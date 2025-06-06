import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import ctypes
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import PIXOR
from loss import CustomLoss

from utils_attack import *
from utils_nuscs import *
from utils import get_bev
from postprocess import compute_iou, convert_format, non_max_suppression
from utils import get_model_name, load_config

from export_kitti import *


def get_gt3Dboxes(dataset_dir, frame_id):
    # get groundtruth bounding box paramethers: class, w, h, l, y, z, x, yaw
    label_path = os.path.join(dataset_dir, 'label_2', frame_id + '.txt')

    object_list = {'car': 1, 'Truck': 0, 'DontCare': 0, 'Van': 0, 'Tram': 0}
    bbox_list = []
    with open(label_path, 'r') as f:
        lines = f.readlines()  # get rid of \n symbol
        for line in lines:
            bbox = []
            entry = line.split(' ')
            name = entry[0]
            if name in list(object_list.keys()):
                bbox.append(object_list[name])
                bbox.extend([float(e) for e in entry[1:]])
                if name == 'car':
                    w, h, l, y, z, x, yaw = bbox[8:15] # hwlxyz
                    y = -y
                    z = -z
                    yaw = -(yaw + np.pi / 2)
                    bbox_list.append([w, h, l, y, z, x, yaw])
    return bbox_list

def build_model(config, device, train=True):
    net = PIXOR(config['geometry'], config['use_bn'])
    loss_fn = CustomLoss(device, config, num_classes=1)

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


def get_attacked_boxes(token: str,
                       gt_box: Box,
                       root: str,
                       attack_det_path: str,
                       max_dist: float = None) -> List[Box]:

    boxes = []
    split_folder, inst_sample_token = token.split('_')

    with open(os.path.join(root, split_folder, 'label_2_attacked', inst_sample_token+'.txt'), 'r') as f:
        for line in f:
            # Parse this line into box information.
            parsed_line = parse_attacked_label_line(line)

            gt_z = gt_box.center[2]
            gt_h = gt_box.wlh[2]

            score = parsed_line['score']
            x = parsed_line['x_lidar']
            y = parsed_line['y_lidar']
            wlh = (parsed_line['w'], parsed_line['l'], gt_h)
            yaw_lidar = parsed_line['yaw_lidar']
            quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_lidar)

            box = Box([x, y, gt_z], wlh, quat_box, name='car')

            # 4: Transform to nuScenes LIDAR coord system.
            kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
            box.rotate(kitti_to_nu_lidar)

            # Set score or NaN.
            box.score = score

            # Set dummy velocity.
            box.velocity = np.array((0.0, 0.0, 0.0))

            # Optional: Filter by max_dist
            if max_dist is not None:
                dist = np.sqrt(np.sum(box.center[:2] ** 2))
                if dist > max_dist:
                    continue

            boxes.append(box)

    return boxes


def lidar_nusc_box_to_global(nusc, boxes, sample_token):

    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))

        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        box_list.append(box)
    return box_list


def show_det_delta(data_dir, scene_name, instance_token):
    # scene_name = os.listdir(data_dir)[0]
    assert args.scene_name.startswith('scene') or args.scene_name == 'all'
    dataset_paths = []
    if args.scene_name == 'all':
        dataset_paths = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.startswith('scene')]
    else:
        if isinstance(instance_token, list):
            for t in instance_token:
                dataset_paths.append(os.path.join(data_dir, scene_name + '-' + t))
        else:
            dataset_paths.append(os.path.join(data_dir, scene_name + '-' + instance_token))

    for dataset_dir in dataset_paths:
        # read attack results
        attack_det_path = os.path.join(dataset_dir, 'attack_det_pred.json')
        with open(attack_det_path, 'r') as f:
            attack_dets_lidar = json.load(f)["results"]
        sample_tokens = attack_dets_lidar.keys()

        frame_ids = sample_tokens

        # create adv det distribution
        adv_det_dir = os.path.join(dataset_dir, 'attack_det_distrib')
        if not os.path.exists(adv_det_dir):
            os.makedirs(adv_det_dir)

        # lidar to global
        attack_det_pred = {}
        for frame_id, data in attack_dets_lidar.items():
            label_path = os.path.join(dataset_dir, 'label_2', frame_id + '.txt')
            with open(label_path, 'r') as f:
                lines = f.readlines()  # get rid of \n symbol
                for line in lines:
                    bbox = []
                    entry = line.split(' ')
                    name = entry[0]
                    if name == 'car':
                        bbox.extend([float(e) for e in entry[1:]])
                        w, h, l, y, z, x, yaw = bbox[7:14]
                        y = -y
                        z = -z
                        yaw = -(yaw + np.pi / 2)

            attack_det_pred[frame_id] = []
            for box in data:
                ### transform target vehicle ###
                # Extract position and heading for target and reference vehicles
                _, target_x, target_y, target_z, target_len, target_wid, target_ht, target_heading = box
                # Store target global pose
                attack_det_pred[frame_id].append([target_x, target_y, target_heading])
                ### target vehicle ###

            ### plot attack det distribution ###
            # get gt label
            x_gt = x
            y_gt = y
            heading_gt = yaw

            # get attack det
            attack_det_np = np.array(attack_det_pred[frame_id])

            # visualize x, y, heading using histogram
            x = attack_det_np[:, 0] - x_gt
            y = attack_det_np[:, 1] - y_gt
            heading = attack_det_np[:, 2] - heading_gt

            plt.figure(figsize=(20, 5))
            plt.subplot(1, 3, 1)
            plt.hist(x, bins=100)
            plt.title('delta x')
            plt.subplot(1, 3, 2)
            plt.hist(y, bins=100)
            plt.title('delta y')
            plt.subplot(1, 3, 3)
            plt.hist(heading, bins=100)
            plt.title('delta heading')
            plt.savefig(os.path.join(adv_det_dir, f'{frame_id}.png'))

def _box_to_sample_result(sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
    # Prepare data
    translation = box.center
    size = box.wlh
    rotation = box.orientation.q
    velocity = box.velocity
    detection_name = box.name
    detection_score = box.score

    # Create result dict
    sample_result = dict()
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = translation.tolist()
    sample_result['size'] = size.tolist()
    sample_result['rotation'] = rotation.tolist()
    sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
    sample_result['detection_name'] = detection_name
    sample_result['detection_score'] = detection_score
    sample_result['attribute_name'] = attribute_name

    return sample_result

def process_res_to_lidar_kitti(res, token, root):
    # Get transforms matrix for this sample
    transforms = get_transforms(token, root=root)

    score, x, y, z, l, w, h, yaw = res
    yaw = -(yaw + np.pi / 2)
    z = -z
    y = -y
    h_cam, w_cam, l_cam, x_cam, y_cam, z_cam, yaw_cam = w, h, l, y, z, x, yaw

    center = (float(x_cam), float(y_cam), float(z_cam))
    wlh = (float(w_cam), float(l_cam), float(h_cam))
    yaw_camera = float(yaw_cam)
    name = 'car'
    score = float(score)

    # 1: Create box in Box coordinate system with center at origin.
    # The second quaternion in yaw_box transforms the coordinate frame from the object frame
    # to KITTI camera frame. The equivalent cannot be naively done afterwards, as it's a rotation
    # around the local object coordinate frame, rather than the camera frame.
    quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(axis=(1, 0, 0), angle=np.pi / 2)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

    # 2: Translate: KITTI defines the box center as the bottom center of the vehicle. We use true center,
    # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(center + np.array([0, -wlh[2] / 2, 0]))

    # 3: Transform to KITTI LIDAR coord system. First transform from rectified camera to camera, then
    # camera to KITTI lidar.
    box.rotate(Quaternion(matrix=transforms['r0_rect']).inverse)
    box.translate(-transforms['velo_to_cam']['T'])
    box.rotate(Quaternion(matrix=transforms['velo_to_cam']['R']).inverse)

    output = {
        'score': score,
        'x_lidar': box.center[0],
        'y_lidar': box.center[1],
        'w': box.wlh[0],
        'l': box.wlh[1],
        'yaw_lidar': box.orientation.yaw_pitch_roll[0],
    }

    return output


def attack(data_dir, net, config, device):
    # init
    scene_names = os.listdir(data_dir)
    scene_names = [s for s in scene_names if s.startswith('scene')]

    attack_dets = {}
    attack_addpts_points = {}

    assert args.scene_name.startswith('scene') or args.scene_name == 'all'

    for s_idx in range(len(scene_names)):
        # init
        dir_name = scene_names[s_idx].split('-')
        scene_name = dir_name[0] + '-' + dir_name[1]
        instance_token = dir_name[2]

        if args.scene_name.startswith('scene') and scene_name != args.scene_name:
            continue

        scene_name = scene_names[s_idx]
        dataset_dir = os.path.join(data_dir, scene_name)

        # load hyper-parameters in config file
        config_path = os.path.join(dataset_dir, 'config.yaml')
        cfg = cfg_from_yaml_file(config_path)
        cfg = cfg.DET

        # get ground truth labels dirs
        sample_tokens = []
        with open(os.path.join(dataset_dir, 'target_sample_tokens.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                sample_token = line.strip()
                sample_tokens.append(sample_token)

        # get attack frame ids
        # TODO: modify the frame_ids to attack
        frame_ids = sample_tokens[args.begin_frame-4: args.begin_frame+1]
        # frame_ids = sample_tokens[6-4: 6+1]

        # get lidar paths
        lidar_paths = [os.path.join(dataset_dir, 'velodyne', f_id + '.bin') for f_id in frame_ids]
        added_points_pool = get_adv_cls(cfg.ATTACK.N_iter, cfg.N_add, cfg.Npts_cls)     # (N_iter,N_add*Npts_cls*4)

        # loop over frames to attack
        for f_idx in range(len(frame_ids)):
            frame_id = frame_ids[f_idx]
            lidar_path = lidar_paths[f_idx]
            label_list = []
            w, h, l, y, z, x, yaw = get_gt3Dboxes(dataset_dir, frame_id)[0]
            # x, y, z, l, w, h, yaw, _, _ = labels_lidar[frame_id]

            ### get ground truth ###
            bev_corners = np.zeros((4, 2), dtype=np.float32)
            # rear left
            bev_corners[0, 0] = x - l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
            bev_corners[0, 1] = y - l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
            # rear right
            bev_corners[1, 0] = x - l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
            bev_corners[1, 1] = y - l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
            # front right
            bev_corners[2, 0] = x + l / 2 * np.cos(yaw) + w / 2 * np.sin(yaw)
            bev_corners[2, 1] = y + l / 2 * np.sin(yaw) - w / 2 * np.cos(yaw)
            # front left
            bev_corners[3, 0] = x + l / 2 * np.cos(yaw) - w / 2 * np.sin(yaw)
            bev_corners[3, 1] = y + l / 2 * np.sin(yaw) + w / 2 * np.cos(yaw)
            geom = config['geometry']['input_shape']
            label_list.append(bev_corners)

            kitti_token = '%s_%s' % (scene_name, frame_id)
            gt_box_lidar = get_gt_box(token=kitti_token, root=data_dir)
            x1 = gt_box_lidar.center[0]
            y1 = gt_box_lidar.center[1]
            z1 = gt_box_lidar.center[2]
            w1 = gt_box_lidar.wlh[0]
            l1 = gt_box_lidar.wlh[1]
            h1 = gt_box_lidar.wlh[2]
            yaw1 = gt_box_lidar.orientation.yaw_pitch_roll[0]
            gt3Dboxes = np.array([[w1, h1, l1, y1, z1, x1, yaw1]])
            ### get ground truth ###

            ### calculate the perturbations ###
            scores = []  # (sample_num, 6)
            for added_points in tqdm(added_points_pool):
                scores.append(attack_obj_nuscs(added_points, net, 0, config, geom, lidar_path, label_list, gt3Dboxes, device, cfg.N_add * cfg.Npts_cls))

            pred = []
            for score in scores:
                temp_s, temp_x, temp_y, temp_w, temp_l, temp_yaw = score
                pred.append([float(temp_s), float(temp_x), float(temp_y), z, float(temp_l), float(temp_w), h, float(temp_yaw)])
            ### calculate the perturbations ###

            ### record detection results ###
            attack_dets[frame_id] = pred
            ### record detection results ###

        # filter nonempty detections
        attack_det_seq = np.zeros((int(cfg.ATTACK.N_iter), 5, 1))  # store scores
        for frame_idx, frame_id in enumerate(frame_ids):
            for box_idx, box_pred in enumerate(attack_dets[frame_id]):
                attack_det_seq[box_idx, frame_idx, 0] = box_pred[1]

        nonempty_idxes = np.where((attack_det_seq[:, :, 0] != 0).all(axis=1))[0]

        # update attack det
        for frame_idx, frame_id in enumerate(frame_ids):
            attack_dets[frame_id] = [attack_dets[frame_id][i] for i in nonempty_idxes]
            print(frame_id, ':', len(attack_dets[frame_id]))

        # record added points
        attack_addpts_points[frame_ids[-1]] = added_points_pool[nonempty_idxes]

        ### save results ###
        # save addpts
        attack_addpts_center_path = os.path.join(dataset_dir, 'attack_addpts_points.pkl')
        print('Writing addpts to: %s' % attack_addpts_center_path)
        with open(attack_addpts_center_path, 'wb') as f:
            pickle.dump(attack_addpts_points, f)

        # save attack_det.json
        # Dummy meta data, please adjust accordingly.
        meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }
        submission = {
            'meta': meta,
            'results': attack_dets
        }
        attack_det_path = os.path.join(dataset_dir, 'attack_det_pred.json')
        print('Writing submission to: %s' % attack_det_path)
        with open(attack_det_path, 'w') as f:
            json.dump(submission, f, indent=2)
        # save results ###

def attack_to_global(data_dir: str,
                     max_dist: float = None,
                     instance_token: str = None):
    assert args.scene_name.startswith('scene') or args.scene_name == 'all'
    dataset_paths = []
    if args.scene_name == 'all':
        dataset_paths = [os.path.join(data_dir, s) for s in os.listdir(data_dir) if s.startswith('scene')]
    else:
        if isinstance(instance_token, list):
            for t in instance_token:
                dataset_paths.append(os.path.join(data_dir, args.scene_name + '-' + t))
        else:
            dataset_paths.append(os.path.join(data_dir, args.scene_name + '-' + instance_token))

    nusc = NuScenes(version='v1.0-trainval', dataroot='mnt/dataset/nuScenes/trainval', verbose=True)
    for dataset_dir in dataset_paths:
        parts = dataset_dir.split('-')
        scene_name = parts[0] + '-' + parts[1]
        instance_token = parts[2]

        target_sample_tokens_path = os.path.join(dataset_dir, 'target_sample_tokens.txt')
        attack_det_path = os.path.join(dataset_dir, 'attack_det_pred.json')

        with open(attack_det_path, 'r') as f:
            attack_det = json.load(f)["results"]

        attack_sample_token = attack_det.keys()

        results = {}
        for sample_token in attack_sample_token:
            kitti_token = '%s_%s' % (scene_name + '-' + instance_token, sample_token)
            gt_box_lidar = get_gt_box(token=kitti_token, root=data_dir)

            ### get attacked boxes in nuscs lidar frame ###
            attack_det_ress = attack_det[sample_token]

            attacked_boxes_lidar = []
            for res in attack_det_ress:
                parsed_line = process_res_to_lidar_kitti(res, kitti_token, data_dir)

                gt_z = gt_box_lidar.center[2]
                gt_h = gt_box_lidar.wlh[2]

                score = parsed_line['score']
                x = parsed_line['x_lidar']
                y = parsed_line['y_lidar']
                wlh = (parsed_line['w'], parsed_line['l'], gt_h)
                yaw_lidar = parsed_line['yaw_lidar']
                quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_lidar)

                box = Box([x, y, gt_z], wlh, quat_box, name='car')

                # 4: Transform to nuScenes LIDAR coord system.
                kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
                box.rotate(kitti_to_nu_lidar)

                # Set score or NaN.
                box.score = score

                # Set dummy velocity.
                box.velocity = np.array((0.0, 0.0, 0.0))

                # Optional: Filter by max_dist
                if max_dist is not None:
                    dist = np.sqrt(np.sum(box.center[:2] ** 2))
                    if dist > max_dist:
                        continue

                attacked_boxes_lidar.append(box)

            ### get gt box in nuScenes LIDAR coord system ###
            kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
            gt_box_lidar.rotate(kitti_to_nu_lidar)


            # Convert nuScenes box in lidar frame to global frame
            boxes = lidar_nusc_box_to_global(nusc, attacked_boxes_lidar, sample_token)
            gt_box_global = lidar_nusc_box_to_global(nusc, [gt_box_lidar], sample_token)[0]

            x = np.array([box.center[0] for box in boxes]) - gt_box_global.center[0]
            y = np.array([box.center[1] for box in boxes]) - gt_box_global.center[1]
            heading = np.array([box.orientation.yaw_pitch_roll[0] for box in boxes]) - \
                      gt_box_global.orientation.yaw_pitch_roll[0]

            # plt.figure(figsize=(20, 5))
            # plt.subplot(1, 3, 1)
            # plt.hist(x, bins=100)
            # plt.title('delta x')
            # plt.subplot(1, 3, 2)
            # plt.hist(y, bins=100)
            # plt.title('delta y')
            # plt.subplot(1, 3, 3)
            # plt.hist(heading, bins=100)
            # plt.title('delta heading')
            # plt.savefig(os.path.join(dataset_dir, 'attack_det_distrib', f'{sample_token}_nuscs_global.png'))

            # # Convert KITTI boxes to nuScenes detection challenge result format.
            inst_sample_results = [_box_to_sample_result(instance_token + '-' + sample_token, box) for box in boxes]

            results[instance_token + '-' + sample_token] = inst_sample_results

        meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        submission = {
            'meta': meta,
            'results': results,
        }
        submission_path = os.path.join(dataset_dir, 'attack_det_nuscs_global.json')
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('--data_dir', default='mnt/dataset/nuScenes/results')
    parser.add_argument('--scene_name', default='scene-0001', help='one specific scene or all')
    parser.add_argument('--begin_frame', default=6, type=int, help='the range of frames to attack is [N-4, N+1]')
    args = parser.parse_args()

    ### init ###
    # init device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    # init config
    config, _, _, _ = load_config('nusce_kitti')

    # init model
    net, loss_fn = build_model(config, device, train=False)
    net.load_state_dict(torch.load(get_model_name(config, epoch=38), map_location=device))
    net.set_decode(True)
    net.eval()
    ### init ###

    # sampling perturbations
    attack(args.data_dir, net, config, device)

    # TODO: modify instance token
    scene_names = os.listdir(args.data_dir)
    scene_names = [s for s in scene_names if s.startswith(args.scene_name)]
    instance_token = scene_names[0].split('-')[2]
    # instance_token = 'b3bd430657754a16b32c7674507dba80'

    # show_det_delta(args.data_dir, args.scene_name, instance_token)
    # Convert nuScenes box in lidar frame to global frame
    attack_to_global(data_dir=args.data_dir, instance_token=instance_token)
