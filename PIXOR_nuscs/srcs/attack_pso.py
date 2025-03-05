import itertools

import pyswarms as ps
import torch.nn as nn
from fastdtw import fastdtw
from numpy.linalg import norm
from scipy.spatial.distance import euclidean

from export_kitti import *
from loss import CustomLoss
from model import PIXOR
from utils import get_model_name, load_config
from utils_attack import *
from utils_attack import center_to_addpts
from utils_nuscs import *


# from .loss import attack_loss
# from .constraint import hard_constraint

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
                    w, h, l, y, z, x, yaw = bbox[8:15]
                    y = -y
                    z = -z
                    yaw = -(yaw + np.pi / 2)
                    bbox_list.append([w, h, l, y, z, x, yaw])
    return bbox_list

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

def get_continue_multi_frame(target_res):
    t1 = target_res[0]
    t2 = target_res[1]
    t3 = target_res[2]
    t4 = target_res[3]
    t5 = target_res[4]
    combinations = list(itertools.product(t1, t2, t3, t4, t5))
    return combinations

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def calculate_D_max(H1, H2):
    distances = np.sum(np.square(H1[:, :2] - H2[:, :2]), axis=1)

    D_max = np.max(distances)
    if D_max == 0:
        D_max = 1e-6

    # D_max = 16
    return D_max

def compute_similarity_loss(det_res, target_res):
    # 1. ade
    D_max = calculate_D_max(det_res, target_res)
    ade = np.sum(np.square(det_res[:, :2] - target_res[:, :2])) / len(target_res)
    ade_norm = ade / D_max
    position_similarity = 1 - ade_norm
    # position_distances = np.mean([euclidean(det_res[i, :2], target_res[i, :2]) for i in range(len(det_res))])

    # 2. cos similarity
    heading_similarities = np.mean([cosine_similarity(det_res[i, 2:], target_res[i, 2:]) for i in range(len(det_res))])

    # 3. DTW
    dtw_distance, _ = fastdtw(det_res[:, :2], target_res[:, :2], dist=euclidean)

    alpha = 0.4  # weight of pose loss
    beta = 0.2   # weight of heading loss
    gamma = 0.4  # weight of shape loss

    similarity_score = alpha * position_similarity + beta * heading_similarities + gamma * (1 - dtw_distance)

    # inverse similarity score to loss
    return 1 - similarity_score


def objective(x_birds, net, config, device, sample_tokens, scene_name, target_res, nusc, data_root):
    data_dir = data_root
    dir_names = os.listdir(data_dir)
    dir_names = [s for s in dir_names if s.startswith(scene_name)]
    dir_name = dir_names[0]
    dataset_dir = os.path.join(data_dir, dir_name)
    config_path = os.path.join(dataset_dir, 'config.yaml')
    cfg = cfg_from_yaml_file(config_path)
    cfg = cfg.DET

    frame_ids = sample_tokens
    lidar_paths = [os.path.join(dataset_dir, 'velodyne', f_id + '.bin') for f_id in frame_ids]

    N = x_birds.shape[0]
    loss = np.zeros(x_birds.shape[0])

    for n in range(N):
        # used for precise optimization
        # added_points = x_birds[n, :].reshape((1, 3*4*4))

        # used for vague optimization
        added_points = x_birds[n, :].reshape((cfg.N_add, 3))  # n
        added_points = center_to_addpts(1, cfg.N_add, cfg.Npts_cls, 0.2, added_points)

        det_res = []

        for f_idx in range(len(frame_ids)):
            frame_id = frame_ids[f_idx]

            lidar_path = lidar_paths[f_idx]
            label_list = []
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
            # reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]
            geom = config['geometry']['input_shape']
            label_list = bev_corners[np.newaxis, :]

            kitti_token = '%s_%s' % (dir_name, frame_id)
            gt_box_lidar = get_gt_box(token=kitti_token, root=data_dir)
            x1 = gt_box_lidar.center[0]
            y1 = gt_box_lidar.center[1]
            z1 = gt_box_lidar.center[2]
            w1 = gt_box_lidar.wlh[0]
            l1 = gt_box_lidar.wlh[1]
            h1 = gt_box_lidar.wlh[2]
            yaw1 = gt_box_lidar.orientation.yaw_pitch_roll[0]
            gt3Dboxes = np.array([[w1, h1, l1, y1, z1, x1, yaw1]])  # Lidar frame system

            score = attack_obj_nuscs(added_points, net, 0, config, geom, lidar_path, label_list, gt3Dboxes, device, cfg.N_add * cfg.Npts_cls)

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

            det_res.append([target_x_global, target_y_global, target_heading_global])

        _loss = compute_similarity_loss(np.array(det_res), target_res)
        loss[n] += _loss

    return loss


def get_bounds_cls(N_add, Npts_cls):
    lower_bound = np.array([-2.1, -2.1, -0.1, 0.4])
    upper_bound = np.array([2.1, 2.1, 0.9, 0.7])
    lower_bound = np.tile(lower_bound, (N_add * Npts_cls, 1)).reshape(N_add*Npts_cls*4)
    upper_bound = np.tile(upper_bound, (N_add * Npts_cls, 1)).reshape(N_add*Npts_cls*4)
    center = (lower_bound + upper_bound) / 2
    return lower_bound, upper_bound, center

def get_bounds_center(N_add):
    lower_bound = np.array([-2., -2., 0.])
    upper_bound = np.array([2., 2., 0.8])
    lower_bound = np.tile(lower_bound, (N_add, 1)).reshape(N_add*3)
    upper_bound = np.tile(upper_bound, (N_add, 1)).reshape(N_add*3)
    center = (lower_bound + upper_bound) / 2
    return lower_bound, upper_bound, center

class BaseAttacker():
    def __init__(self, obs_length, pred_length, net, config, device):
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.net = net
        self.config = config
        self.device = device

class PSOAttacker(BaseAttacker):
    def __init__(self, obs_length, pred_length, net, config, device, nusc, data_dir, n_particles=10, iter_num=10, c1=2, c2=2, w=1.0, bound=np.array([2, 2, 0.8]), physical_bounds={}):
        super().__init__(obs_length, pred_length, net, config, device)
        self.iter_num = iter_num
        self.bound = bound
        self.options = {'c1': c1, 'c2': c2, 'w': w}
        self.n_particles = n_particles
        self.nusc = nusc
        self.data_dir = data_dir

    def run(self, sample_tokens, scene_name, target_res):
        target_res = target_res
        lower_bound, upper_bound, center = get_bounds_center(3)
        ### vague optimization ###
        optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=3 * 3, options=self.options,
                                            bounds=(lower_bound, upper_bound),
                                            center=center)
        ### vague optimization ###

        ### test the presice optimization ###
        # lower_bound, upper_bound, center = get_bounds_cls(3, 4)
        # optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=3 * 16, options=self.options,
        #                                     bounds=(lower_bound, upper_bound),
        #                                     center=center)
        ### test the presice optimization ###

        best_loss, best_perturb = optimizer.optimize(objective, iters=self.iter_num, net=self.net, config=self.config,
                                                     device=self.device, sample_tokens=sample_tokens,
                                                     scene_name=scene_name, target_res=target_res, nusc=self.nusc, data_root=self.data_dir)

        return best_loss, best_perturb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnt/dataset/nuScenes/results')
    parser.add_argument('--scene_name', type=str, default='scene-0043')
    parser.add_argument('--target_per_dir', type=int, default='mnt/dataset/nuScenes/results/scene-0043-011d7348763d4841859209e9aeab6a2a')
    parser.add_argument('--nusc_datadir', type=str, default='mnt/dataset/nuScenes/trainval')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval')
    parser.add_argument('--begin_frame', type=int, default=12)
    args = parser.parse_args()

    scene_name = args.scene_name
    multi_frame_target_path = os.path.join(args.target_per_dir, 'inverse', 'inverse_res_multi_frame.json')
    # multi_frame_target_path = 'mnt/dataset/nuScenes/results/scene-0043-011d7348763d4841859209e9aeab6a2a/inverse/inverse_res_multi_frame.json'
    with open(multi_frame_target_path, 'r') as f:
        multi_frame_target = json.load(f)
    f.close()

    target_r = []
    for i in range(1,6):
        k = scene_name + '_{}'.format(str(i))
        target_res = multi_frame_target[k]
        pos = target_res['pos'][:3]
        heading = target_res['heading'][:3]
        res = np.concatenate((pos, heading), axis=1)
        target_r.append(res)
    target_results = get_continue_multi_frame(target_r)     # get continue target perturbation

    # get attack frame ids
    sample_tokens = []
    with open(os.path.join(args.target_per_dir, 'target_sample_tokens.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sample_token = line.strip()
            sample_tokens.append(sample_token)
    f.close()
    sample_tokens = sample_tokens[args.begin_frame - 4: args.begin_frame + 1]  # todo: modify the range of attack frame ids

    data_dir = args.data_dir

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

    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_datadir, verbose=True)

    loss = []
    perturb = []
    for i in range(len(target_results)):
        res = np.zeros((5, 3))
        temp = target_results[i]
        for j in range(5):
            res[j] = temp[j]
        print('Step :', i)
        # pso for the best pos(x,y,z)
        attacker = PSOAttacker(4, 6, net, config, device, nusc, data_dir, n_particles=10)
        best_loss, best_perturb =attacker.run(sample_tokens, scene_name, res)
        loss.append(best_loss)
        perturb.append(best_perturb.tolist())

    pso_res = {
        scene_name:{
            'loss': loss,
            'perturb': perturb,
        }
    }

    # save pso_res
    current_path = os.path.join(args.target_per_dir, 'inverse')
    # current_path = 'mnt/dataset/nuScenes/results/scene-0043-011d7348763d4841859209e9aeab6a2a/inverse/'
    with open(os.path.join(current_path, 'pso_res.json'), 'w') as f:
        json.dump(pso_res, f)




