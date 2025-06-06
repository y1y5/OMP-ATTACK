import yaml
from easydict import EasyDict

import sys
import torch

sys.path.append('../../trajectron')
from helper import *
from ftocp import *
from environment import derivative_of
from pyquaternion import Quaternion
import math

def find_straight_segment(trajectory, threshold=1):
    n_frames = trajectory.shape[0]
    best_start, best_end = 0, 0
    best_length = 0

    for start in range(n_frames):
        for end in range(start + 2, n_frames):
            start_point = trajectory[start]
            end_point = trajectory[end]

            vec = end_point - start_point
            vec = vec / np.linalg.norm(vec)
            distances = np.abs(np.cross(trajectory[start:end + 1] - start_point, vec))

            if np.all(distances < threshold):
                length = end - start
                if length > best_length:
                    best_length = length
                    best_start, best_end = start, end

    return best_start, best_end

def merge_new_config(config, new_config):

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file):

    config = EasyDict()

    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


### loss ###
def ade(predicted_trajs, gt_traj):
    
    error = torch.norm(predicted_trajs - gt_traj, dim=-1)
    ade = torch.mean(error, dim=-1)
    
    return ade

def min_ade(predicted_trajs, gt_traj):
    
    error = torch.norm(predicted_trajs - gt_traj, dim=-1)
    ade = torch.min(error)
    
    return ade

def mse(predict_trajs, future_trajs):

    return torch.sum(torch.square(predict_trajs - future_trajs)) / predict_trajs.shape[0]


def weighted_mse(predict_trace, future_trace, w=torch.tensor([1., 1., 1., 1., 1., 1.]).cuda().reshape(6, 1)):

    return torch.sum(w * torch.square(predict_trace - future_trace)) / predict_trace.shape[0]


def loss_matching(curr_perturb, cluster_perturb):
    return torch.sum(torch.abs(curr_perturb - cluster_perturb))

### loss ###


### sampling techniques ###
def sample_ego_states(ego_state, sample_scale=2.0, dt=0.5):

    print('use sample_ego_states')
    raise ValueError()

    # Initialize the state of the vehicle
    ego_state_sim = np.zeros_like(ego_state)

    # Set the state at frame t
    ego_state_sim[-1] = ego_state[-1]

    # Compute the state at frames t-1, t-2, t-3, t-4 using the equations of motion in reverse
    for t in range(4, 0, -1):
        # Sample a change in acceleration and heading
        delta_a = np.random.normal(0, sample_scale*np.linalg.norm(ego_state[t, 4:6]))  # 10% standard deviation
        delta_yaw = np.random.normal(0, sample_scale*abs(ego_state[t, 6]))  # 10% standard deviation

        # Compute the state at frame t-1
        ego_state_sim[t-1, 6] = ego_state_sim[t, 6] - delta_yaw

        ego_state_sim[t-1, 4:6] = ego_state_sim[t, 4:6] - delta_a
        ego_state_sim[t-1, 2:4] = ego_state_sim[t, 2:4] - ego_state_sim[t, 4:6]*dt
        ego_state_sim[t-1, 0:2] = ego_state_sim[t, 0:2] - ego_state_sim[t, 2:4]*dt + 0.5*ego_state_sim[t, 4:6]*dt**2

    # change delta heading accordingly
    # Note: derivative should only apply non nan frames, or the output will be all 0.
    ego_state_sim[:,-1] = derivative_of(ego_state_sim[:, 6], dt, radian=True)

    return ego_state_sim

### sampling techniques ###


### plan ###
def plan(predict_trajs, scene, target_instance_id, ts_curr):
    # init planner
    M = 1
    ft = 6
    ts_range_curr = np.array([ts_curr])
    ts_range_pred = np.array([ts_curr+1, ts_curr+6])

    planner = FTOCP(ft, M, scene.dt, scene.robot.width, scene.robot.length)

    # init inputs
    # set nodes
    nodes = []
    if isinstance(target_instance_id, list):
        for n in scene.nodes:
            if n.id in target_instance_id:
                nodes.append(n)
                if len(nodes) == len(target_instance_id):
                    break
    else:
        for n in scene.nodes:
            if n.id == target_instance_id:
                nodes.append(n)
                break
    
    # set x0 (ego states at current ts)
    x0_states = {'position': ['x', 'y'], 'velocity': ['norm'], 'heading': ['°']}
    x0_vel = {'velocity': ['x', 'y']}
    for n in scene.nodes:
        if n.id == 'ego':
            x0 = n.get(tr_scene=ts_range_curr, state=x0_states)[0]
            x0_vel = n.get(tr_scene=ts_range_curr, state=x0_vel)[0]  # to generate xref
            break

    # set xref
    # # 1. use ego's ground truth future states as xref
    # xref_states = {'position': ['x', 'y'], 'velocity': ['norm'], 'heading': ['°']}
    # for n in scene.nodes:
    #     if n.id == 'ego':
    #         xref = n.get(tr_scene=ts_range_pred, state=xref_states)
    #         # print(xref)
    #         break
    # 2. use ego's estimated future states as xref
    x0_x, x0_y, x0_vnorm, x0_h = (x0[0], x0[1], x0[2], x0[3])
    x0_vx, x0_vy = (x0_vel[0], x0_vel[1])
    x_d = (
        x0_vx * scene.dt * np.arange(ft).reshape([ft, 1])
    )
    y_d = (
        x0_vy * scene.dt * np.arange(ft).reshape([ft, 1])
    )
    xref = np.hstack(
        (
            x0_x + x_d,
            x0_y + y_d,
            x0_vnorm * np.ones([ft, 1]),
            x0_h * np.ones([ft, 1])
        )
    )
    # print('xref:', xref)

    if isinstance(target_instance_id, list):
        ypreds = []
        for i in range(len(target_instance_id)):
            ypreds.append([predict_trajs[target_instance_id[i]].detach().cpu().numpy()])
    else:
        ypreds = [[predict_trajs[target_instance_id].detach().cpu().numpy()]]
    
    # set w
    w = np.array([1.])
    
    # plan
    planner.buildandsolve(nodes, x0, xref, ypreds, w)
    xplan = planner.xSol[1:].reshape((M, ft, 4))

    xplan_scene = xplan[0][:, :2]
    
    return xplan_scene


def plan_multivelo(predict_trajs, scene, target_instance_id, ts_curr, x0, x0_vel):
    # init planner
    M = 1
    ft = 6
    ts_range_curr = np.array([ts_curr])
    ts_range_pred = np.array([ts_curr+1, ts_curr+6])

    planner = FTOCP(ft, M, scene.dt, scene.robot.width, scene.robot.length)

    # init inputs
    # set nodes
    nodes = []
    for n in scene.nodes:
        if n.id == target_instance_id:
            nodes.append(n)
            break

    # 2. use ego's estimated future states as xref
    x0_x, x0_y, x0_vnorm, x0_h = (x0[0], x0[1], x0[2], x0[3])
    x0_vx, x0_vy = (x0_vel[0], x0_vel[1])
    x_d = (
        x0_vx * scene.dt * np.arange(ft).reshape([ft, 1])
    )
    y_d = (
        x0_vy * scene.dt * np.arange(ft).reshape([ft, 1])
    )
    xref = np.hstack(
        (
            x0_x + x_d,
            x0_y + y_d,
            x0_vnorm * np.ones([ft, 1]),
            x0_h * np.ones([ft, 1])
        )
    )
    # print('xref:', xref)

    # set ypreds (shape [[(6, 4)], [(6, 4)], ...])
    # 1. pack x, y, v, h into ypreds
    # ypreds_states = {'position': ['x', 'y'], 'heading': ['°']}
    # for n in scene.nodes:
    #     if n.id == target_instance_id:
    #         target_curr = n.get(tr_scene=ts_range_curr, state=ypreds_states)
    #         break
    # target_curr_xy = target_curr[:, :2]

    # ypreds_xy = np.concatenate((target_curr_xy, predict_trajs[target_instance_id].detach().cpu().numpy()), axis=0)
    # vx = derivative_of(ypreds_xy[:, 0], scene.dt)
    # vy = derivative_of(ypreds_xy[:, 1], scene.dt)
    # ypreds_v = np.linalg.norm(np.stack((vx, vy), axis=1), axis=1)[1:].reshape((6, 1))
    # # estimate heading
    # ypreds_h = np.zeros((6, 1))
    # for i in range(1, 6):
    #     ypreds_h[i, 0] = np.arctan2(ypreds_xy[i, 1] - ypreds_xy[i-1, 1], ypreds_xy[i, 0] - ypreds_xy[i-1, 0])
    # # concate xyvh
    # ypreds = [[np.concatenate((ypreds_xy[1:], ypreds_v, ypreds_h), axis=1)]]
    # 2. pack x, y into ypreds
    ypreds = [[predict_trajs[target_instance_id].detach().cpu().numpy()]]
    # print('ypreds:', ypreds)
    # raise ValueError()
    
    # set w
    w = np.array([1.])
    
    # plan
    planner.buildandsolve(nodes, x0, xref, ypreds, w)
    xplan = planner.xSol[1:].reshape((M, ft, 4))

    # x_min, y_min, _, _ = scene.patch
    # xplan = xplan[0][:, :2] + np.array([x_min, y_min])

    xplan_scene = xplan[0][:, :2]
    
    return xplan_scene
### plan ###


### output format ###
def out_format(data):
    # Base case: if data is already a list
    if isinstance(data, list):
        return [out_format(item) for item in data]

    if isinstance(data, torch.Tensor):  # If data is a PyTorch tensor
        return data.tolist()
    
    if isinstance(data, np.ndarray):  # If data is a NumPy array
        return data.tolist()
    
    return data
### output format ###