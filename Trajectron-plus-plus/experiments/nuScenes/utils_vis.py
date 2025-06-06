import sys
sys.path.append('../../trajectron')
from helper import *
from utils import prediction_output_to_trajectories

layers = ['drivable_area',
        'road_segment',
        'lane',
        'ped_crossing',
        'walkway',
        'stop_line',
        'road_divider',
        'lane_divider']

def rotate_trajectory_to_y_axis(trajectory, heading):
    angle = np.pi / 2 - heading
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_trajectory = np.dot(trajectory, rotation_matrix.T)
    return rotated_trajectory

### visualization ###
robot = plt.imread('icons/Car TOP_VIEW 80CBE5.png')
# adversary = plt.imread('icons/Car TOP_VIEW ROBOT.png')
adversary = plt.imread('icons/Car TOP_VIEW ABCB51.png')  # green
normal_vehicle = plt.imread('icons/Car TOP_VIEW 375397.png') # blue

def draw_common(predictions, target_instance_id, scene, ego_cur_rot, output_folder, output_fn, egpplan_attack, ph=6, normal_pred=None, normal_gt=None, nor_av_rot=None, vic_future_draw=False):
    fig, ax = plt.subplots(1, figsize=(12, 12))  # (10, 10)

    my_patch = scene.patch
    x_min, y_min, x_max, y_max = my_patch
    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(predictions,
                                                                                      scene.dt,
                                                                                      5,
                                                                                      ph,
                                                                                      map=None)
    assert (len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)
    vic_node = None
    adv_node = None
    for node in node_list:
        if node.id == target_instance_id:
            adv_node = node
        elif node.id == 'ego':
            vic_node = node

    vic_history = histories_dict[vic_node] + np.array([x_min, y_min])
    vic_future = futures_dict[vic_node] + np.array([x_min, y_min])
    vic_predictions = prediction_dict[vic_node] + np.array([x_min, y_min])
    vic_plan_attack = egpplan_attack
    vic_plan_attack = vic_plan_attack.reshape(-1, 2)
    # vic_plan_atk = np.concatenate([vic_history[-1:, :2], vic_plan_atk], axis=0)
    vic_future = vic_future.reshape(-1, 2)
    vic_future = np.concatenate([vic_history[-1:, :2], vic_future], axis=0)

    adv_history = histories_dict[adv_node] + np.array([x_min, y_min])
    adv_future = futures_dict[adv_node] + np.array([x_min, y_min])
    adv_predictions = prediction_dict[adv_node] + np.array([x_min, y_min])
    adv_predictions = adv_predictions.reshape(-1, 2)

    # add current pos
    vic_plan_attack = np.concatenate([vic_history[-1:, :2], vic_plan_attack], axis=0)
    adv_predictions = np.concatenate([adv_history[-1:, :2], adv_predictions], axis=0)

    vic_plan_attack = rotate_trajectory_to_y_axis(vic_plan_attack, ego_cur_rot)
    adv_predictions = rotate_trajectory_to_y_axis(adv_predictions, ego_cur_rot)
    vic_future = rotate_trajectory_to_y_axis(vic_future, ego_cur_rot)
    if normal_pred is not None:
        normal_pred = rotate_trajectory_to_y_axis(normal_pred, ego_cur_rot)
    if normal_gt is not None:
        normal_gt = rotate_trajectory_to_y_axis(normal_gt, ego_cur_rot)


    # # Center vic_tra around x = 0
    x_origin = vic_plan_attack[0, 0]
    y_origin = vic_plan_attack[0, 1]
    vic_plan_attack[:] = vic_plan_attack[:] - np.array([x_origin, y_origin])
    adv_predictions[:] = adv_predictions[:] - np.array([x_origin, y_origin])
    vic_future[:] = vic_future[:] - np.array([x_origin, y_origin])
    if normal_pred is not None:
        normal_pred[:] = normal_pred[:] - np.array([x_origin, y_origin])
    if normal_gt is not None:
        normal_gt[:] = normal_gt[:] - np.array([x_origin, y_origin])

    if normal_pred is None and normal_gt is None:
        x_min = np.min(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0])))
        x_max = np.max(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0])))
        y_min = np.min(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1])))
        y_max = np.max(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1])))
    elif normal_pred is not None and normal_gt is None:
        x_min = np.min(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_pred[:, 0])))
        x_max = np.max(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_pred[:, 0])))
        y_min = np.min(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_pred[:, 1])))
        y_max = np.max(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_pred[:, 1])))
    elif normal_pred is None and normal_gt is not None:
        x_min = np.min(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_gt[:, 0])))
        x_max = np.max(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_gt[:, 0])))
        y_min = np.min(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_gt[:, 1])))
        y_max = np.max(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_gt[:, 1])))
    else:
        x_min = np.min(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_pred[:, 0], normal_gt[:, 0])))
        x_max = np.max(np.concatenate((vic_plan_attack[:, 0], adv_predictions[:, 0], normal_pred[:, 0], normal_gt[:, 0])))
        y_min = np.min(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_pred[:, 1], normal_gt[:, 1])))
        y_max = np.max(np.concatenate((vic_plan_attack[:, 1], adv_predictions[:, 1], normal_pred[:, 1], normal_gt[:, 1])))


    # Plot victim vehicle
    vic_img = rotate(robot, 90, reshape=True)
    oi_vic = OffsetImage(vic_img, zoom=0.044)
    veh_box_vic = AnnotationBbox(oi_vic, (vic_plan_attack[0, 0], vic_plan_attack[0, 1]), frameon=False)
    veh_box_vic.zorder = 1  # Higher z-order
    ax.add_artist(veh_box_vic)

    # Plot adversarial vehicle
    vic_img = rotate(adversary, 90, reshape=True)
    oi_vic = OffsetImage(vic_img, zoom=0.044)
    veh_box_vic = AnnotationBbox(oi_vic, (adv_predictions[0, 0], adv_predictions[0, 1]), frameon=False)
    veh_box_vic.zorder = 1  # Higher z-order
    ax.add_artist(veh_box_vic)

    # Plot normal vehicle
    if normal_pred is not None or normal_gt is not None:
        vic_img = rotate(normal_vehicle, nor_av_rot * 180 / np.pi + 90, reshape=True)
        oi_vic = OffsetImage(vic_img, zoom=0.022)
        veh_box_vic = AnnotationBbox(oi_vic, (normal_pred[0, 0], normal_pred[0, 1]), frameon=False)
        veh_box_vic.zorder = 1  # Higher z-order
        ax.add_artist(veh_box_vic)

    # Plot victim (vic_tra)
    ax.scatter(vic_plan_attack[0, 0], vic_plan_attack[0, 1], color='blue', marker='*', s=500, label='Victim current')
    # ax.plot(vic_plan_attack[:, 0], vic_plan_attack[:, 1], 'g^', label='Victim plan', markersize=8)
    for i in range(len(vic_plan_attack) - 1):
        ax.quiver(vic_plan_attack[i, 0], vic_plan_attack[i, 1], vic_plan_attack[i + 1, 0] - vic_plan_attack[i, 0], vic_plan_attack[i + 1, 1] - vic_plan_attack[i, 1],
                   angles='xy', scale_units='xy', scale=1, color='green', label='Victim Plan' if i == 0 else "")


    # Plot victim (vic_future)
    if vic_future_draw:
        # ax.plot(vic_plan_attack[:, 0], vic_plan_attack[:, 1], 'g^', label='Victim plan', markersize=8)
        for i in range(len(vic_future) - 1):
            ax.quiver(vic_future[i, 0], vic_future[i, 1], vic_future[i + 1, 0] - vic_future[i, 0],
                    vic_future[i + 1, 1] - vic_future[i, 1],
                    angles='xy', scale_units='xy', scale=1, color='blue', label='Victim Future' if i == 0 else "")

    # Plot adversary (adv_tra)
    # plt.plot(adv_tra[:, 0], adv_tra[:, 1], 'r-', label='Adv history')
    ax.scatter(adv_predictions[0, 0], adv_predictions[0, 1], color='red', marker='*', s=500, label='Adv current')
    # ax.plot(adv_predictions[:, 0], adv_predictions[:, 1], 'y^', label='Adv predict', linewidth=8)
    for i in range(len(adv_predictions) - 1):
        ax.quiver(adv_predictions[i, 0], adv_predictions[i, 1], adv_predictions[i + 1, 0] - adv_predictions[i, 0], adv_predictions[i + 1, 1] - adv_predictions[i, 1],
                  color='orange', scale_units='xy', angles='xy', scale=1, label='Adversarial Predict' if i == 0 else "")

    # Plot normal vehicle (normal_tra)
    if normal_pred is not None or normal_gt is not None:
        ax.scatter(normal_pred[0, 0], normal_pred[0, 1], color='yellow', marker='*', s=300, label='Normal current')
        if normal_pred is not None:
            for i in range(len(normal_pred) - 1):
                ax.quiver(normal_pred[i, 0], normal_pred[i, 1], normal_pred[i + 1, 0] - normal_pred[i, 0], normal_pred[i + 1, 1] - normal_pred[i, 1],
                          color='yellow', scale_units='xy', angles='xy', scale=1, label='Normal Predict' if i == 0 else "")
        if normal_gt is not None:
            for i in range(len(normal_gt) - 1):
                ax.quiver(normal_gt[i, 0], normal_gt[i, 1], normal_gt[i + 1, 0] - normal_gt[i, 0], normal_gt[i + 1, 1] - normal_gt[i, 1],
                          color='green', scale_units='xy', angles='xy', scale=1, label='Normal GT' if i == 0 else "")

    fs = 25
    ax.legend(fontsize=fs)

    # Add text annotations for vehicles
    ax.text(adv_predictions[0, 0] - 3, adv_predictions[0, 1] - 4, 'Adversarial vehicle', fontsize=fs)
    ax.text(vic_plan_attack[0, 0] + 1, vic_plan_attack[0, 1] - 0.5, 'Victim AV', fontsize=fs)
    if normal_pred is not None or normal_gt is not None:
        ax.text(normal_pred[0, 0] + 1, normal_pred[0, 1] - 1, 'Normal vehicle', fontsize=fs)

    # Set plot limits and labels
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)

    ax.set_xlim([x_min - 2, x_max + 2])
    ax.set_ylim([y_min - 2, y_max + 2])
    ax.set_xlabel('X (m)', fontsize=fs)
    ax.set_ylabel('Y (m)', fontsize=fs)
    fig.savefig(os.path.join(output_folder, output_fn), dpi=300, bbox_inches='tight')
    plt.clf()  # close to save memory


def draw_pred(predictions, target_instance_id, scene, nusc_map, output_folder, output_fn, egoplan_attack=None, egoplan_clean=None, mode='mm', ph=6, other_vehicle=None):
    # mode: 
    # - 'mm': draw most likely prediction
    # - 'dist': draw a distribution of predictions
    
    # load corresponding map
    my_patch = scene.patch
    x_min, y_min, x_max, y_max = my_patch
    vis_patch = scene.vis_patch
    x_min_vis, y_min_vis, x_max_vis, y_max_vis = vis_patch
    # print(vis_patch)
    fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(10, 10), alpha=0.1, render_egoposes_range=False)

    # plot predictions (scene)
    if mode == 'mm':
        plot_attacked_vehicle_mm(ax,
                    predictions,
                    scene.dt,
                    max_hl=5,
                    ph=ph,
                    map=None, x_min=x_min, y_min=y_min,
                    vis_mode='ego_adv',
                    target_instance_id=target_instance_id,
                    other_vehicle=other_vehicle)
    elif mode == 'dist':
        plot_attacked_vehicle_nice(ax,
            predictions,
            scene.dt,
            max_hl=10,
            ph=ph,
            map=None, x_min=x_min, y_min=y_min,
            target_instance_id=target_instance_id)
    else:
        raise ValueError('mode {} not supported'.format(mode))
    
    # ego plan (global)
    if egoplan_attack is not None:
        ax.plot(egoplan_attack[:, 0], egoplan_attack[:, 1], 'ro-',
        zorder=700,
        markersize=3,
        linewidth=3, alpha=0.7)

    if egoplan_clean is not None:
        ax.plot(egoplan_clean[:, 0], egoplan_clean[:, 1], 'bo-',
        zorder=650,
        markersize=3,
        linewidth=3, alpha=0.7, label='Clean') # 650

    # save figure
    ax.set_xlim((x_min_vis, x_max_vis))
    ax.set_ylim((y_min_vis, y_max_vis))
    leg = ax.legend(loc='upper right', fontsize=20, frameon=True)
    ax.axis('off')
    for lh in leg.legendHandles:
        lh.set_alpha(.5)    
    ax.get_legend().remove()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    fig.savefig(os.path.join(output_folder, output_fn), dpi=300, bbox_inches='tight')
    
    plt.clf()  # close to save memory
### visualization ###


### PGP visualization ###
import torch
import math

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

def angle_of_rotation(yaw):
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (math.pi / 2) + torch.sign(-yaw) * torch.abs(yaw)


def global_to_local(global_coord, origin_trans, origin_yaw):

    yaw = angle_of_rotation(origin_yaw)

    transform = torch.tensor([[torch.cos(yaw), torch.sin(yaw)],
                                [-torch.sin(yaw), torch.cos(yaw)]], dtype=global_coord.dtype).to(global_coord.device)

    # coords = (global_coord - torch.tensor(origin_trans[:2], dtype=global_coord.dtype).unsqueeze(0)).t()
    coords = (global_coord - origin_trans[:2].clone().detach().unsqueeze(0)).t()

    return torch.mm(transform, coords).t()[:, :2]

def plot(tar_pred, tar_pred_atk, vic_plan, vic_plan_atk, i_t, s_t, predicthelper, out_fn):

    # Raster maps for visualization
    # map_extent = [-50, 50, -20, 80]  # left, right, behind, ahead
    map_extent = [-30, 30, -20, 30]
    resolution = 0.1
    static_layer_rasterizer = StaticLayerRasterizer(predicthelper,
                                                    resolution=resolution,
                                                    meters_ahead=map_extent[3],
                                                    meters_behind=-map_extent[2],
                                                    meters_left=-map_extent[0],
                                                    meters_right=map_extent[1])

    agent_rasterizer = AgentBoxesWithFadedHistory(predicthelper, seconds_of_history=1,
                                                    resolution=resolution,
                                                    meters_ahead=map_extent[3],
                                                    meters_behind=-map_extent[2],
                                                    meters_left=-map_extent[0],
                                                    meters_right=map_extent[1])

    raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    # Get raster map
    hd_map = raster_maps.make_input_representation(i_t, s_t)
    r, g, b = hd_map[:, :, 0] / 255, hd_map[:, :, 1] / 255, hd_map[:, :, 2] / 255
    hd_map_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # plot
    fig, ax = plt.subplots(1, figsize=(5, 5))
    # ax.imshow(hd_map, cmap=[r,g,b], extent=map_extent)
    ax.imshow(hd_map, cmap='gist_gray', extent=map_extent)
    # ax.imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)
    # ax[2].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)
    
    # plot most likely prediction
    ax.plot(tar_pred[:6, 0].detach().cpu().numpy(), tar_pred[:6, 1].detach().cpu().numpy(), lw=2, color='g', alpha=0.8)
    ax.scatter(tar_pred[5, 0].detach().cpu().numpy(), tar_pred[5, 1].detach().cpu().numpy(), 60, color='g', alpha=0.8)

    # plot most likely prediction under attack
    ax.plot(tar_pred_atk[:6, 0].detach().cpu().numpy(), tar_pred_atk[:6, 1].detach().cpu().numpy(), lw=2, color='r', alpha=0.8)
    ax.scatter(tar_pred_atk[5, 0].detach().cpu().numpy(), tar_pred_atk[5, 1].detach().cpu().numpy(), 60, color='r', alpha=0.8)
    
    # plot ego plan
    vic_plan = vic_plan.detach().cpu().numpy()
    ax.plot(vic_plan[:6, 0], vic_plan[:6, 1], lw=2, color='g')
    ax.scatter(vic_plan[5, 0], vic_plan[5, 1], 60, color='g')

    # plot ego plan under attack
    vic_plan_atk = vic_plan_atk.detach().cpu().numpy()
    ax.plot(vic_plan_atk[:6, 0], vic_plan_atk[:6, 1], lw=2, color='r')
    ax.scatter(vic_plan_atk[5, 0], vic_plan_atk[5, 1], 60, color='r')

    ax.axis('off')
    # ax[1].axis('off')
    # ax[2].axis('off')
    # ax[3].axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    # ax[1].margins(0)
    # ax[2].margins(0)
    # ax[3].margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.savefig('{}'.format(out_fn), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


robot = plt.imread('icons/Car TOP_VIEW 80CBE5.png')
# adversary = plt.imread('icons/Car TOP_VIEW ROBOT.png')
adversary = plt.imread('icons/Car TOP_VIEW ABCB51.png')  # green
# adversary = plt.imread('icons/Car TOP_VIEW C8B0B0.png')  # brown

def plot_paper(tar_pred, tar_pred_atk, tar_his_atk,
            vic_plan, vic_plan_atk, vic_his_atk, i_t, s_t, predicthelper, out_fn):

    # Raster maps for visualization
    # map_extent = [-50, 50, -20, 80]  # left, right, behind, ahead
    # map_extent = [-17.5, 3, -25, 20]  # acc-891
    map_extent = [-13, 2.5, -20, 10] # brake
    # map_extent = [-17.5, 3, -25, 25] # lane-change
    vehicle_size = 0.022
    resolution = 0.1
    static_layer_rasterizer = StaticLayerRasterizer(predicthelper,
                                                    resolution=resolution,
                                                    meters_ahead=map_extent[3],
                                                    meters_behind=-map_extent[2],
                                                    meters_left=-map_extent[0],
                                                    meters_right=map_extent[1],
                                                    colors=[(255, 255, 255), (119, 136, 153), (255, 255, 255)])

    agent_rasterizer = AgentBoxesWithFadedHistory(predicthelper, seconds_of_history=1,
                                                    resolution=resolution,
                                                    meters_ahead=map_extent[3],
                                                    meters_behind=-map_extent[2],
                                                    meters_left=-map_extent[0],
                                                    meters_right=map_extent[1])

    raster_maps = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    # Get raster map
    hd_map = raster_maps.make_input_representation(i_t, s_t)
    r, g, b = hd_map[:, :, 0] / 255, hd_map[:, :, 1] / 255, hd_map[:, :, 2] / 255
    hd_map_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # plot
    fig, ax = plt.subplots(1, figsize=(5, 5))
    # ax.imshow(hd_map, extent=map_extent)
    ax.imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)
    # ax[2].imshow(hd_map_gray, cmap='gist_gray', extent=map_extent)

    # settings
    linewid = 2

    # victim vehicle's history trajectory
    vic_his_atk = vic_his_atk.detach().cpu().numpy()
    ax.plot(vic_his_atk[:, 0], vic_his_atk[:, 1], 'b-', label='Victim History', zorder=20, linewidth=linewid)
    alphas = np.linspace(0.2, 0.8, vic_his_atk.shape[0])
    for i in range(vic_his_atk.shape[0]):
        ax.plot(vic_his_atk[i, 0], vic_his_atk[i, 1], 'bo', alpha=alphas[i], zorder=21, linewidth=linewid)
    ax.plot(vic_his_atk[-1, 0], vic_his_atk[-1, 1], 'b*', label='Victim Current', markersize=10, zorder=30)  # Blue star for victim

    # Victim vehicle's planned future trajectory
    vic_plan_atk = vic_plan_atk.detach().cpu().numpy()
    vic_plan_atk = np.concatenate([vic_his_atk[-1:, :2], vic_plan_atk], axis=0)
    ax.quiver(vic_plan_atk[:-1, 0], vic_plan_atk[:-1, 1],
              vic_plan_atk[1:, 0] - vic_plan_atk[:-1, 0],
              vic_plan_atk[1:, 1] - vic_plan_atk[:-1, 1],
              color='green', scale_units='xy', angles='xy', scale=1, label='Victim Future', linestyle='dashed', zorder=20, width=0.02)

    # Adversarial vehicle's history trajectory
    tar_his_atk = tar_his_atk.detach().cpu().numpy()
    ax.plot(tar_his_atk[:, 0], tar_his_atk[:, 1], 'r-', label='Adv History', zorder=20)
    alphas = np.linspace(0.2, 0.8, tar_his_atk.shape[0])  # Gradual increase from 0.1 to 1
    for i in range(tar_his_atk.shape[0]):
        ax.plot(tar_his_atk[i, 0], tar_his_atk[i, 1], 'ro', alpha=alphas[i], zorder=21)
    ax.plot(tar_his_atk[-1, 0], tar_his_atk[-1, 1], 'r*', label='Adv Current', markersize=10, zorder=30)  # Red star for adversary

    # Adversarial vehicle's predicted future trajectory
    tar_pred_atk = tar_pred_atk.detach().cpu().numpy()
    tar_pred_atk = np.concatenate([tar_his_atk[-1:, :2], tar_pred_atk], axis=0)
    ax.quiver(tar_pred_atk[:-1, 0], tar_pred_atk[:-1, 1],
              tar_pred_atk[1:, 0] - tar_pred_atk[:-1, 0],
              tar_pred_atk[1:, 1] - tar_pred_atk[:-1, 1],
              color='orange', scale_units='xy', angles='xy', scale=1, label='Adv Future', linestyle='dashed', zorder=20, width=0.03)
    
    # Plot adversarial vehicle
    adv_img = rotate(adversary, 90, reshape=True) # 90 degrees for y-axis alignment
    oi_adv = OffsetImage(adv_img, zoom=vehicle_size) # half size
    veh_box_adv = AnnotationBbox(oi_adv, (0, 0), frameon=False)
    # veh_box_adv = AnnotationBbox(oi_adv, (tar_his_atk[-1, 0], tar_his_atk[-1, 1]), frameon=False)
    # veh_box_adv = AnnotationBbox(oi_adv, (tar_his_atk[-1, 0]+1, tar_his_atk[-1, 1]+0.4), frameon=False)  # acc
    veh_box_adv.zorder = 10  # Lower z-order
    ax.add_artist(veh_box_adv)

    # Plot victim vehicle
    vic_img = rotate(robot, 90, reshape=True) # 90 degrees for y-axis alignment
    oi_vic = OffsetImage(vic_img, zoom=vehicle_size)
    veh_box_vic = AnnotationBbox(oi_vic, (vic_his_atk[-1, 0], vic_his_atk[-1, 1]), frameon=False)
    veh_box_vic.zorder = 10  # Higher z-order
    ax.add_artist(veh_box_vic)

    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)

    # fig.legend(fontsize='15')

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.savefig('{}'.format(out_fn), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
### PGP visualization ###

from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw


def get_target_agent_yaw(helper, i_t, s_t):

    sample_annotation = helper.get_sample_annotation(i_t, s_t)
    yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
    yaw = correct_yaw(yaw)

    return yaw