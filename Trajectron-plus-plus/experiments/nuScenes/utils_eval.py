import os
import json
import math
import torch
import pickle
import numpy as np

import csv
import datetime


### metric: collision ###
# check whether ego vehicle's center is inside the ellipse
def check_collision(pos_ego, pos_adv, dim_ego, dim_adv):
    for i in range(6):
        
        c = (pos_ego[i, 0] - pos_adv[i, 0])**2 / (dim_ego[0]/1.414 + dim_adv[0]/1.414)**2 + \
            (pos_ego[i, 1] - pos_adv[i, 1])**2 / (dim_ego[1]/1.414 + dim_adv[1]/1.414)**2
        
        if c < 1:
            return True

    return False
### metric: collision ###


### metric: lateral/horizontal deviation ### 
def horizonal_distance(observe_trace, predict_trace, future_trace):
    offset = predict_trace - future_trace

    direction = (future_trace - 
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)), 
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    right_direction = torch.matmul(
                        torch.tensor([[0., 1.], [-1., 0.]]).float().to("cuda"),
                        direction.t().float() / scale).t()

    # average_distance = torch.sum(offset * right_direction) / predict_trace.shape[0]
    # print('rotated offset: ', offset * right_direction)
    distances = torch.sum(offset * right_direction, dim=1)
    
    return distances


def vertical_distance(observe_trace, predict_trace, future_trace):
    offset = predict_trace - future_trace
    direction = (future_trace - 
                 torch.cat(
                   (torch.reshape(observe_trace[-1,:], (1,2)), 
                    future_trace[:-1,:]), 0)).float()
    scale = torch.sqrt(torch.sum(torch.square(direction), 1)).float()
    
    # average_distance = torch.sum(offset * (direction.t().float() / scale).t()) / predict_trace.shape[0]
    distances = torch.sum(offset * (direction.t().float() / scale).t(), dim=1)
    
    return distances
### metric: lateral/horizontal deviation ### 


###  metric: lateral/longtitude acceleration ### 
def get_unit_vector(vectors):
    scale = np.sum(vectors ** 2, axis=1) ** 0.5 + 0.001
    result = np.zeros(vectors.shape)
    result[:,0] = vectors[:,0] / scale
    result[:,1] = vectors[:,1] / scale
    return result

def get_acceleration(observe_traj, plan_traj, dt=0.5):

    trace_array = torch.cat((torch.reshape(observe_traj[-5:,:], (5,2)), 
                             plan_traj), 0).detach().cpu().numpy()
    
    v = (trace_array[1:,:] - trace_array[:-1,:]) / dt
    a = (v[1:,:] - v[:-1,:]) / dt

    direction = get_unit_vector(v)
    direction_r = np.concatenate((direction[:,1].reshape(direction.shape[0],1), 
                                -direction[:,0].reshape(direction.shape[0],1)), axis=1)

    long_a = np.sum(direction[:-1,:] * a, axis=1)
    lat_a = np.sum(direction_r[:-1,:] * a, axis=1)

    return long_a, lat_a
###  metric: lateral/longtitude acceleration ### 


def get_attack_res(attack_res_path, target_scene_id, target_instance_id):

    with open(attack_res_path, 'rb') as f:
        attack_res_dict = pickle.load(f)
    
    scene_inst = 'scene-{:04d}-{}'.format(target_scene_id, target_instance_id)
    attack_res = attack_res_dict[scene_inst]

    dh_set = attack_res['dh']
    dp_set = attack_res['dp']
    heading_set = attack_res['heading']
    pos_set = attack_res['pos']

    return dh_set, dp_set, heading_set, pos_set


def get_attack_res_json(attack_res_path, target_scene_id, target_instance_id):

    with open(attack_res_path, 'rb') as f:
        attack_res_dict = json.load(f)
    
    scene_inst = 'scene-{:04d}-{}'.format(target_scene_id, target_instance_id)
    attack_res = attack_res_dict[scene_inst]

    def list_to_tensor(list):
        return [torch.tensor(data).float().cuda() for data in list]

    dh_set = list_to_tensor(attack_res['dh'])
    dp_set = list_to_tensor(attack_res['dp'])
    heading_set = list_to_tensor(attack_res['heading'])
    pos_set = list_to_tensor(attack_res['pos'])

    return dh_set, dp_set, heading_set, pos_set


def get_eval_frames_20hz(pos_20hz_det, heading_20hz_det, ego_pose_20hz, key_flag_20hz, velo_scale):
    # pos_20hz_det: pytorch tensor (80,2)
    # heading_20hz_det: pytorch tensor (80,1)
    # ego_pose_20hz: pytorch tensor (80,3) x, y, heading
    # velo_scale: float range in [0.1, 2]

    # get default frames based on key_flag_20hz
    # get indexes of 1s in key_flag_20hz
    default_frames = [i for i in range(len(key_flag_20hz)) if key_flag_20hz[i] == 1]
    last_frame_idx = default_frames[-1]
    default_frames = default_frames[:5]  # (0, ..., )

    # calc frame interval based on velo_scale (original interval: 0.5s, 20hz)    
    eval_frames = [0] + [int(i*velo_scale) for i in default_frames[1:]]  # (0, ....)
    eval_frames.reverse()  # (..., 0)

    # process empty or out-of-range frames
    for i in range(len(eval_frames)):

        frame_idx = eval_frames[i]

        # check whether frame_idx is out of range
        if frame_idx >= last_frame_idx:
            eval_frames[i] = last_frame_idx
            continue

        # check whether all columns in pos_20hz_det[frame_idx, :] is zero
        if torch.sum(pos_20hz_det[frame_idx, :]) == 0 and torch.sum(heading_20hz_det[frame_idx, :]) == 0:

            ego_pose = ego_pose_20hz[frame_idx, :][:2]

            # find the non-zero frame that has the cloest ego pose to the zero frame
            min_dist = 999.
            min_idx = None
            for j in range(pos_20hz_det.shape[0]):
                if j == frame_idx:
                    continue
                
                if torch.sum(pos_20hz_det[j, :]) != 0 and torch.sum(heading_20hz_det[j, :]) != 0:
                    ego_pose_temp = ego_pose_20hz[j, :][:2]
                    dist = torch.norm(ego_pose - ego_pose_temp)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j

            # print('empty {}: change to {}'.format(frame_idx, min_idx))
            eval_frames[i] = min_idx
    
    return eval_frames



### kalman filter ###
import numpy as np
from kalman_filter import NonlinearKinematicBicycle


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x

def derivative_of(x, dt=1., radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx

def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def ekf_tracker(cand, scene_dt=0.5):

    # init
    x = cand['delta_position'][:, 0].cpu().numpy()  # (5,)
    y = cand['delta_position'][:, 1].cpu().numpy()  # (5,)
    heading = cand['delta_heading'][:, 0].cpu().numpy()  # (5,)

    # Compute Velocity
    vx = derivative_of(x, scene_dt)
    vy = derivative_of(y, scene_dt)
    velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

    # Kalman Filter for Nonlinear Kinematic Bicycle
    filter_veh = NonlinearKinematicBicycle(dt=scene_dt, sMeasurement=1.0)
    P_matrix = None
    for i in range(len(x)):
        if i == 0:  # initialize KF
            # initial P_matrix
            P_matrix = np.identity(4)
        elif i < len(x) - 1:
            # assign new estimated values
            x[i] = x_vec_est_new[0][0]
            y[i] = x_vec_est_new[1][0]
            heading[i] = x_vec_est_new[2][0]
            velocity[i] = x_vec_est_new[3][0]

        if i < len(x) - 1:  # no action on last data
            # filtering
            x_vec_est = np.array([[x[i]],
                                  [y[i]],
                                  [heading[i]],
                                  [velocity[i]]])
            z_new = np.array([[x[i + 1]],
                              [y[i + 1]],
                              [heading[i + 1]],
                              [velocity[i + 1]]])
            x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                x_vec_est=x_vec_est,
                u_vec=np.array([[0.], [0.]]),
                P_matrix=P_matrix,
                z_new=z_new
            )
            P_matrix = P_matrix_new


    # format output
    cand['delta_position'][:, 0] = torch.tensor(x[-5:]).cuda()
    cand['delta_position'][:, 1] = torch.tensor(y[-5:]).cuda()
    cand['delta_heading'][:, 0] = torch.tensor(heading[-5:]).cuda()

    return cand
# ### kalman filter ###
        

### output ###
def output_to_csv(eval_res: dict, output_path: str, atk_type='forward', query=100, velo=1, tag=''):

    # calc metrics
    collision_num = 0
    pi_ade_sum = 0.
    pr_ade_sum = 0.
    lat_dev_sum = 0.
    long_dev_sum = 0.
    lat_jerk_sum = 0.
    long_jerk_sum = 0.

    for k, v in eval_res.items():
        pi_ade_sum += v['pi_ade']
        pr_ade_sum += v['pr_ade']
        if v['is_collision']:
            collision_num += 1

        lat_dev_sum += abs(v['lat_dev'])
        long_dev_sum += abs(v['long_dev'])
        lat_jerk_sum += abs(v['lat_jerk'])
        long_jerk_sum += abs(v['long_jerk'])

    pi_ade = pi_ade_sum / len(eval_res)
    pr_ade = pr_ade_sum / len(eval_res)
    collision_rate = collision_num / len(eval_res)
    lat_dev = lat_dev_sum / len(eval_res)
    long_dev = long_dev_sum / len(eval_res)
    lat_jerk = lat_jerk_sum / len(eval_res)
    long_jerk = long_jerk_sum / len(eval_res)

    # check whether csv exists
    if not os.path.exists(output_path):
        # create csv
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            header = ['timestep', 'type', 'query', 'velo', 'tag', 'piade', 'prade',
                      'cr', 'lat_dev', 'long_dev', 'lat_jerk', 'long_jerk']
            writer.writerow(header)
            
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    # init current time in str format (month, day, hour, minute)
    now = datetime.datetime.now()
    timestep = '{}/{} {}:{}'.format(str(now.month), str(now.day), str(now.hour), str(now.minute))
    row = [timestep, atk_type, query, velo, tag, pi_ade, pr_ade, collision_rate, lat_dev, long_dev, lat_jerk, long_jerk]

    # check if row with same type, query, velo, and tag exists
    existing_row = None
    for r in rows[1:]:  # skip header
        if r[1] == atk_type and r[2] == str(query) and r[3] == str(velo) and r[4] == tag:
            existing_row = r
            break

    if existing_row:
        # update metrics in the existing row
        existing_row[0] = timestep
        existing_row[5:] = row[5:]
    else:
        # append the new row
        rows.append(row)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print('output to csv: {}'.format(output_path))
### output ###


### defense ###
def check_heading_feasibility(vehicle_states, wheelbase=2.5, max_steering_angle=np.radians(60), dt=0.5):
    """
    Check if heading changes are feasible based on the kinematic bicycle model.

    Parameters:
    - vehicle_states: A (5, 3) numpy array with x, y, heading for 5 frames.
    - wheelbase: The wheelbase of the vehicle.
    - max_steering_angle: Maximum steering angle (in radians).
    - dt: Time interval between frames.

    Returns:
    - Boolean indicating if the heading changes are feasible.
    """
    exceedance_count = 0  # Counter for exceedances

    for i in range(1, vehicle_states.shape[0]):
        # Assume constant speed for simplicity
        speed = np.sqrt((vehicle_states[i, 0] - vehicle_states[i-1, 0])**2 + (vehicle_states[i, 1] - vehicle_states[i-1, 1])**2) / dt
        expected_delta_heading = speed * dt * np.tan(max_steering_angle) / wheelbase
        
        actual_delta_heading = abs(vehicle_states[i, 2] - vehicle_states[i-1, 2])
        
        # If the actual heading change exceeds what's expected, it's suspicious
        if actual_delta_heading > expected_delta_heading:
            exceedance_count += 1
            
            if exceedance_count >= 2:
                return False

    return True
### defense ###





