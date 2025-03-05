import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np

np.set_printoptions(suppress=True, precision=3, threshold=2000, linewidth=150)
import argparse
from torch.multiprocessing import Pool, set_start_method

from loss import CustomLoss
# from model_test import PIXOR
from model import PIXOR
from tqdm import tqdm
from utils import get_model_name, load_config
# from postprocess import filter_pred, compute_matches, compute_ap, compute_iou, convert_format

from utils_nuscs import *
from utils_attack import *

from attack_sampling import get_gt3Dboxes, process_res_to_lidar_kitti, lidar_nusc_box_to_global
from export_kitti import get_gt_box

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper


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


def custom_loss(xyh, xyh_goal, xyh_gt, alpha=1.0, beta=0.2):
    # Ensure xyh and xyh_goal are numpy arrays
    xyh = np.array(xyh).squeeze()[:2]
    xyh_goal = np.array(xyh_goal).squeeze()[:2]
    xyh_gt = np.array(xyh_gt).squeeze()[:2]

    # Value Loss
    value_loss = np.abs(xyh - xyh_goal)

    # perturbation sign loss
    perturb_sign_loss = np.sign(xyh - xyh_gt)
    perturb_sign_loss_goal = np.sign(xyh_goal - xyh_gt)
    perturb_sign_loss = np.maximum(0, -perturb_sign_loss * perturb_sign_loss_goal)

    # Total Loss
    # total_loss = alpha * np.sum(sign_loss) + beta * np.sum(value_loss)
    total_loss = alpha * np.sum(value_loss) + beta * np.sum(perturb_sign_loss)
    # print(total_loss, alpha * np.sum(value_loss), beta * np.sum(perturb_sign_loss))
    # raise ValueError()

    return total_loss


def get_gt_boxes3d_lidar(label_path):
    with open(label_path, 'rb') as f:
        gt_box3d_lidar = pickle.load(f)

    x, y, z = gt_box3d_lidar.center
    w, l, h = gt_box3d_lidar.wlh
    yaw = gt_box3d_lidar.orientation.yaw_pitch_roll[0]

    return x, y, z, l, w, h, yaw

def eval(data_dir, config, device):
    # init
    scene_names = os.listdir(data_dir)
    scene_names = [s for s in scene_names if s.startswith('scene')]

    eval_res = {}

    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_datadir, verbose=True)

    for s_idx in range(len(scene_names)):

        # init
        dir_name = scene_names[s_idx].split('-')
        scene_name = dir_name[0] + '-' + dir_name[1]
        instance_token = dir_name[2]

        scene_inst_name = scene_names[s_idx]
        if args.scene_name.startswith('scene') and scene_name != args.scene_name:
            continue
        dataset_dir = os.path.join(data_dir, scene_inst_name)
        curr_work_dir = os.path.join(dataset_dir, 'inverse')

        # load hyper-parameters
        config_path = os.path.join(dataset_dir, 'config.yaml')
        cfg = cfg_from_yaml_file(config_path)
        cfg = cfg.DET

        # get attack frame ids
        sample_tokens = []
        with open(os.path.join(dataset_dir, 'target_sample_tokens.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                sample_token = line.strip()
                sample_tokens.append(sample_token)
        f.close()
        # todo: modify the range of frame_ids
        frame_ids = sample_tokens[args.begin_frame - 4: args.begin_frame + 1]
        # frame_ids = sample_tokens[8 - 4: 8 + 1]

        # get lidar paths
        lidar_paths = [os.path.join(dataset_dir, 'velodyne', f_id + '.bin') for f_id in frame_ids]

        # load pso points
        with open(os.path.join(curr_work_dir, 'pso_res.json'), 'r') as f:
            pso_res_pool = json.load(f)[scene_name]
            loss_pool = pso_res_pool['loss']
            perturb_pool = pso_res_pool['perturb']

            top_k = 3
            top_k_idx = np.argsort(loss_pool)[:top_k]
            # top_k_idx = np.argsort(loss_pool)[-3:]
            loss_pool = [loss_pool[i] for i in top_k_idx]
            perturb_pool = [perturb_pool[i] for i in top_k_idx]

            addpts_points_refined_pool = []
            for i in range(len(loss_pool)):
                perturb_pos = np.array(perturb_pool[i])

                perturb_pos = perturb_pos.reshape((cfg.N_add, 3))
                for j in range(20):
                    addpts_points_refined = center_to_addpts(N_iter=1, N_add=cfg.N_add,
                                                            Npts_cls=cfg.Npts_cls, size_cls=0.2, added_points_center=perturb_pos)

                    addpts_points_refined_pool.append(addpts_points_refined[0].tolist())

        pos_pool = []
        heading_pool = []
        for pool_idx in range(len(addpts_points_refined_pool)):

            pos = []
            heading = []
            for f_idx in range(len(frame_ids)):
                frame_id = frame_ids[f_idx]

                lidar_path = lidar_paths[f_idx]
                # x, y, z, l, w, h, yaw, _, _ = labels_lidar[frame_id]
                w, h, l, y, z, x, yaw = get_gt3Dboxes(dataset_dir, frame_id)[0]

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
                label_list = bev_corners[np.newaxis, :]

                # gt3Dboxes = np.array([[w, h, l, y, z, x, yaw]])
                kitti_token = '%s_%s' % (scene_inst_name, frame_id)
                gt_box_lidar = get_gt_box(token=kitti_token, root=data_dir)
                x1 = gt_box_lidar.center[0]
                y1 = gt_box_lidar.center[1]
                z1 = gt_box_lidar.center[2]
                w1 = gt_box_lidar.wlh[0]
                l1 = gt_box_lidar.wlh[1]
                h1 = gt_box_lidar.wlh[2]
                yaw1 = gt_box_lidar.orientation.yaw_pitch_roll[0]
                gt3Dboxes = np.array([[w1, h1, l1, y1, z1, x1, yaw1]])

                ### attack ###
                addpts_points_refined = addpts_points_refined_pool[pool_idx]  # (N_add, 3)
                addpts_points_refined = np.array(addpts_points_refined)

                score = attack_obj_nuscs(addpts_points_refined, net, 0, config, geom, lidar_path, label_list,
                                             gt3Dboxes, device, cfg.N_add * cfg.Npts_cls)
                ### attack ###

                ### convert to global frame ###
                s, x, y, w, l, yaw = score
                res = [float(s), float(x), float(y), z, float(l), float(w), h, float(yaw)]
                parsed_line = process_res_to_lidar_kitti(res, kitti_token, data_dir)
                x = parsed_line['x_lidar']
                y = parsed_line['y_lidar']
                wlh = (parsed_line['w'], parsed_line['l'], 0)
                yaw_lidar = parsed_line['yaw_lidar']
                quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_lidar)
                box = Box([x, y, 0], wlh, quat_box, name='car')
                # 4: Transform to nuScenes LIDAR coord system.
                kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
                box.rotate(kitti_to_nu_lidar)

                box_global = lidar_nusc_box_to_global(nusc, [box], frame_id)[0]
                target_x_global, target_y_global, target_heading_global = box_global.center[0], box_global.center[1], \
                box_global.orientation.yaw_pitch_roll[0]

                # Store target global pose
                pos.append([target_x_global, target_y_global])
                heading.append([target_heading_global])
                ### convert to global frame ###

            pos_pool.append(pos)
            heading_pool.append(heading)

        eval_res[scene_name] = {
            'pos': pos_pool,
            'heading': heading_pool,
        }

    with open(os.path.join(dataset_dir, 'infer_det_pso_res.json'), 'w') as f:
        json.dump(eval_res, f)
    print('Eval results saved to', os.path.join(dataset_dir, 'infer_det_pso_res.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIXOR custom implementation')
    parser.add_argument('--data_dir', default='mnt/dataset/nuScenes/results')
    parser.add_argument('--scene_name', default='scene-0103')
    parser.add_argument('--nusc_datadir', type=str, default='mnt/dataset/nuScenes/trainval')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    parser.add_argument('--begin_frame', type=int)
    parser.add_argument('--stage', default='infer')
    args = parser.parse_args()

    ### init ###
    # init device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)  # set the default GPU device to use
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

    eval(args.data_dir, config, device)
