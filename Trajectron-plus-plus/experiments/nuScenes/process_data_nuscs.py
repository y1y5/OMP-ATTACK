import sys
import os
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import NonlinearKinematicBicycle
from sklearn.model_selection import train_test_split

nu_path = './devkit/python-sdk/'
# sys.path.append(nu_path)
sys.path.insert(0, nu_path)
sys.path.append("../../trajectron")
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from environment import Environment, Scene, Node, GeometricMap, derivative_of

# scene_blacklist = [499, 515, 517]

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def process_scene(ns_scene, env, nusc, data_path, target_inst_list):
    scene_id = int(ns_scene['name'].replace('scene-', ''))
    data = pd.DataFrame(columns=['frame_id',
                                 'type',
                                 'node_id',
                                 'robot',
                                 'x', 'y', 'z',
                                 'length',
                                 'width',
                                 'height',
                                 'heading'])

    sample_token = ns_scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    frame_id = 0
    while sample['next']:
        annotation_tokens = sample['anns']
        for annotation_token in annotation_tokens:
            annotation = nusc.get('sample_annotation', annotation_token)
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                continue

            # keep only moving and parked vehicles
            if 'vehicle' in category and ('parked' in attribute or 'moving' in attribute or 'stopped' in attribute) and \
                'bicycle' not in category and 'motorcycle' not in category:
                our_category = env.NodeType.VEHICLE
            else:
                continue

            data_point = pd.Series({'frame_id': frame_id,
                                    'type': our_category,
                                    'node_id': annotation['instance_token'],
                                    'robot': False,
                                    'x': annotation['translation'][0],
                                    'y': annotation['translation'][1],
                                    'z': annotation['translation'][2],
                                    'width': annotation['size'][0],
                                    'length': annotation['size'][1],
                                    'height': annotation['size'][2],
                                    'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0]})
            data = data.append(data_point, ignore_index=True)
            # data = pd.concat([data, data_point], ignore_index=True)

        # Ego Vehicle
        our_category = env.NodeType.VEHICLE
        sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])

        data_point = pd.Series({'frame_id': frame_id,
                                'type': our_category,
                                'node_id': 'ego',
                                'robot': True,
                                'x': annotation['translation'][0],
                                'y': annotation['translation'][1],
                                'z': annotation['translation'][2],
                                'length': 4,
                                'width': 1.7,
                                'height': 1.5,
                                'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                'orientation': None})

        data = data.append(data_point, ignore_index=True)
        # data = pd.concat([data, data_point], ignore_index=True)

        sample = nusc.get('sample', sample['next'])
        frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    # get dataframe data with node_id equal to ego
    vis_data = data[data['node_id'] == 'ego']
    vis_x_min = np.round(vis_data['x'].min() - 25)
    vis_x_max = np.round(vis_data['x'].max() + 25)
    vis_y_min = np.round(vis_data['y'].min() - 25)
    vis_y_max = np.round(vis_data['y'].max() + 25)
    vis_patch = (vis_x_min, vis_y_min, vis_x_max, vis_y_max)

    x_min = np.round(data['x'].min() - 50)
    x_max = np.round(data['x'].max() + 50)
    y_min = np.round(data['y'].min() - 50)
    y_max = np.round(data['y'].max() + 50)
    patch = (x_min, y_min, x_max, y_max)

    data['x'] = data['x'] - x_min
    data['y'] = data['y'] - y_min

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), patch=patch, vis_patch=vis_patch)

    # Generate Maps
    map_name = nusc.get('log', ns_scene['log_token'])['location']
    nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)

    type_map = dict()
    x_size = x_max - x_min
    y_size = y_max - y_min
    patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)

    patch_angle = 0  # Default orientation where North is up
    canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
    homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                   'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
    map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
        np.uint8)
    map_mask = np.swapaxes(map_mask, 1, 2)  # x axis comes first
    # PEDESTRIANS
    map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=0)
    type_map['PEDESTRIAN'] = GeometricMap(data=map_mask_pedestrian, homography=homography, description=', '.join(layer_names))
    # VEHICLES
    map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=0)
    type_map['VEHICLE'] = GeometricMap(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

    map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
        max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=0)
    type_map['VISUALIZATION'] = GeometricMap(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

    scene.map = type_map
    del map_mask
    del map_mask_pedestrian
    del map_mask_vehicle
    del map_mask_plot

    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        # uncomment to keep non-continuous nodes (for prediction attack)
        if not np.all(np.diff(node_df['frame_id']) == 1):
            if node_id in target_inst_list:
                print('non-continuous target instance: scene-{:04d}-{}'.format(scene_id, node_id))
            # print('Occlusion')
            target_scene_inst_blacklist.append('scene-{:04d}-{}'.format(scene_id, node_id))
            continue

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, length=node_df.iloc[0]['length'], \
                    width=node_df.iloc[0]['width'], frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]

        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node

        scene.nodes.append(node)
    
    return scene


def process_data(data_path, version, output_path, scn_name):

    # create env
    env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    scenes = []
    
    # process attack scenes
    for ns_scene_name in tqdm(attack_scene_names):

        if not ns_scene_name == scn_name:
            continue

        ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0])
        scene = process_scene(ns_scene, env, nusc, data_path, scene2inst_dict[ns_scene_name])
        if scene is not None:
            scenes.append(scene)

    print(f'Processed {len(scenes):.2f} scenes')

    env.scenes = scenes

    if len(scenes) > 0:
        mini_string = ''
        if 'mini' in version:
            mini_string = '_mini'

        data_dict_path = os.path.join(output_path, 'nuScenes_' + 'attack' + mini_string + '_{}.pkl'.format(scn_name))
        with open(data_dict_path, 'wb') as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        print('Saved Environment!')

    global total
    global curv_0_2
    global curv_0_1
    print(f"Total Nodes: {total}")
    print(f"Curvature > 0.1 Nodes: {curv_0_1}")
    print(f"Curvature > 0.2 Nodes: {curv_0_2}")
    total = 0
    curv_0_1 = 0
    curv_0_2 = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnt/dataset/nuScenes/trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--output_path', type=str, default='mnt/dataset/nuScenes/results/attack_pred/')
    parser.add_argument('--target_scene_instance_file', type=str,
                        default='mnt/dataset/nuScenes/results/attack_pred/scene_inst_tokens.txt')  # todo: need to specify the scene and instance
    parser.add_argument('--scene_name', type=str, default='scene-0043')
    args = parser.parse_args()

    ### init ###
    nusc = NuScenes(version=args.version, dataroot=args.data, verbose=True)

    # read selected scenarios
    target_scene_inst_blacklist = []
    scene_inst_token_path = args.target_scene_instance_file
    # scene_inst_token_path = 'mnt/dataset/nuScenes/results/attack_pred/scene_inst_tokens.txt'
    with open(scene_inst_token_path, 'r') as f:
        target_scene_inst_tokens = f.read().splitlines()
    target_scene_inst_tokens.sort(key=lambda x: (x.split('-')[1], x.split('-')[2]))
    
    # combine scene_inst with same scene name
    scene2inst_dict = {}
    for target_scene_inst_token in target_scene_inst_tokens:
        k = 'scene-'+target_scene_inst_token.split('-')[1] # scene name
        v = target_scene_inst_token.split('-')[2]  # instance token

        if k in scene2inst_dict.keys():
            scene2inst_dict[k].append(v)
        else:
            scene2inst_dict[k] = [v]
    attack_scene_names = list(scene2inst_dict.keys())

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    ### init ###

    # pre-process
    process_data(args.data, args.version, args.output_path, args.scene_name)
