import os
import torch
import dill
import time
import pickle
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from torch.autograd import Variable

from utils_attack import *
from utils_eval import *
from utils_vis import *

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True, precision=3)
import warnings

warnings.filterwarnings('ignore')

def get_target_inst_token(scene_name, scene_inst_tokens_path):
    target_inst_tokens = []
    with open(scene_inst_tokens_path, 'r') as f:
        target_scene_inst_tokens = f.read().splitlines()

    for t in target_scene_inst_tokens:
        scene_t = '-'.join(t.split('-')[0:2])
        if scene_t == scene_name:
            target_inst_tokens.append(t.split('-')[-1])

    return target_inst_tokens

def process():
    dh_set_det_dict = {}  # delta of heading
    dp_set_det_dict = {}  # delta of position

    for scene in eval_scenes:

        ### step 1: init ###
        # scene_name = scene.name
        scene_name = 'scene-' + '0' * (4-len(str(scene.name))) + str(scene.name)
        if args.scene_name.startswith('scene-') and scene_name != args.scene_name:
            continue
        x_min, y_min, _, _ = scene.patch  # scene <-> global

        ts_curr = 4
        timestep = np.array([ts_curr])

        target_inst_tokens_path = os.path.join(args.data_dir, 'attack_pred', 'scene_inst_tokens.txt')
        target_inst_tokens = get_target_inst_token(scene_name, target_inst_tokens_path)

        for target_inst_token in target_inst_tokens:
            dataset_dir = os.path.join(args.data_dir, scene_name + '-' + target_inst_token)
            curr_work_dir = os.path.join(dataset_dir, 'inverse')
            if not os.path.exists(curr_work_dir):
                os.makedirs(curr_work_dir)
            ### step 1: init ###

            ### step 2: load ###
            # load detections
            with open(os.path.join(dataset_dir, 'attack_det_nuscs_global.json'), 'rb') as f:
                attack_det_dict = json.load(f)['results']
            f.close()

            frame_ids = list(attack_det_dict.keys())
            # frame_ids.sort()
            curr_frame_id = frame_ids[ts_curr]

            # get the idx of curr_frame_id in the target node
            with open(os.path.join(dataset_dir, 'target_sample_tokens.txt'), 'r') as f:
                sample_tokens = f.read().splitlines()
            f.close()
            global ts_node
            ts_node = sample_tokens.index(curr_frame_id.split('-')[-1]) # 14

            del sample_tokens

            # load gt
            for i, n in enumerate(scene.nodes):
                if n.id == target_inst_token:
                    ts_range = np.array([ts_node - 4 + n.first_timestep, ts_node + n.first_timestep])
                    box_p_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, :2] + np.array(
                        [x_min, y_min])  # (5, 2) global frame coord
                    box_h_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, 6:7]  # (5, 1)

                    box_p_gt = torch.tensor(box_p_gt).float().cuda()  # double -> float
                    box_h_gt = torch.tensor(box_h_gt).float().cuda()  # double -> float
                    break
            ### step 2: load ###

            ### step 3: calc perturbations ###
            valid_attack_num = len(attack_det_dict[curr_frame_id])
            dh_set_det = torch.zeros((valid_attack_num, 5, 1)).cuda()
            dp_set_det = torch.zeros((valid_attack_num, 5, 2)).cuda()

            for frame_idx, frame_id in enumerate(frame_ids[:5]):

                for det_idx in range(valid_attack_num):
                    box_pred = attack_det_dict[frame_id][det_idx]
                    # box_x, box_y, box_h = box_pred[0], box_pred[1], box_pred[2]
                    box_x, box_y, box_h = box_pred['translation'][0], box_pred['translation'][1], Quaternion(box_pred['rotation']).yaw_pitch_roll[0]

                    dh_set_det[det_idx, frame_idx, :] = torch.tensor([box_h - box_h_gt[frame_idx, 0]]).cuda()
                    dp_set_det[det_idx, frame_idx, :] = torch.tensor(
                        [box_x - box_p_gt[frame_idx, 0], box_y - box_p_gt[frame_idx, 1]]).cuda()
            ### step 3: calc perturbations ###

            dh_set_det_dict[scene_name] = dh_set_det.detach().cpu().numpy()
            dp_set_det_dict[scene_name] = dp_set_det.detach().cpu().numpy()

            # save
            with open(os.path.join(curr_work_dir, 'dh_set_det.pkl'), 'wb') as f:
                pickle.dump(dh_set_det_dict, f)
            with open(os.path.join(curr_work_dir, 'dp_set_det.pkl'), 'wb') as f:
                pickle.dump(dp_set_det_dict, f)


def attack():
    inverse_res = {}

    for scene in eval_scenes:

        ### step 1: init ###

        ### pre-processing scene ###
        # scene_name = scene.name
        scene_name = 'scene-' + '0' * (4 - len(str(scene.name))) + str(scene.name)
        if args.scene_name.startswith('scene-') and scene_name != args.scene_name:
            continue
        x_min, y_min, _, _ = scene.patch  # scene <-> global

        ts_curr = 4
        timestep = np.array([ts_curr])

        target_inst_tokens_path = os.path.join(args.data_dir, 'attack_pred', 'scene_inst_tokens.txt')
        target_inst_token = get_target_inst_token(scene_name, target_inst_tokens_path)[0]

        dataset_dir = os.path.join(args.data_dir, scene_name + '-' + target_inst_token)
        curr_work_dir = os.path.join(dataset_dir, 'inverse')
        if not os.path.exists(curr_work_dir):
            os.makedirs(curr_work_dir)

        config_path = os.path.join(dataset_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise ValueError('config.yaml not exists!')
        cfg = cfg_from_yaml_file(config_path)
        cfg = cfg.PRED


        # load local bias of current frame
        with open(os.path.join(dataset_dir, 'attack_det_nuscs_global.json'), 'rb') as f:
            attack_det_dict = json.load(f)['results']
        f.close()
        frame_ids = list(attack_det_dict.keys())
        # frame_ids.sort()
        curr_frame_id = frame_ids[ts_curr]
        # get the idx of curr_frame_id in the target node
        with open(os.path.join(dataset_dir, 'target_sample_tokens.txt'), 'r') as f:
            sample_tokens = f.read().splitlines()
        f.close()
        global ts_node
        ts_node = sample_tokens.index(curr_frame_id.split('-')[-1])  # 14
        print('ts_node:', ts_node)
        del sample_tokens
        del attack_det_dict
        del frame_ids
        del curr_frame_id

        # load ground truth
        for i, n in enumerate(scene.nodes):
            if n.id == target_inst_token:
                ts_range = np.array([ts_node - 4 + n.first_timestep, ts_node + n.first_timestep])
                box_p_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, :2] + np.array(
                    [x_min, y_min])  # (5, 2) global frame coord
                box_h_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, 6:7]  # (5, 1)

                box_p_gt = torch.tensor(box_p_gt).float().cuda()  # double -> float
                box_h_gt = torch.tensor(box_h_gt).float().cuda()  # double -> float
                timestep = timestep + ts_node + n.first_timestep - 4
                print('timestep:', timestep)
                break

        # load detection perturbation set
        with open(os.path.join(curr_work_dir, 'dh_set_det.pkl'), 'rb') as f:
            dh_set_det_dict = pickle.load(f)
        with open(os.path.join(curr_work_dir, 'dp_set_det.pkl'), 'rb') as f:
            dp_set_det_dict = pickle.load(f)

        dh_set_det = torch.tensor(dh_set_det_dict[scene_name]).float().cuda()
        dp_set_det = torch.tensor(dp_set_det_dict[scene_name]).float().cuda()

        origin_timestep = timestep

        # loop each historical frame
        for attack_idx in [-1, -2, -3, -4, -5]:
            # attack_idx is the attack frame idx
            timestep = origin_timestep + 1 + attack_idx
            # load ground truth
            for i, n in enumerate(scene.nodes):
                if n.id == target_inst_token:
                    ts_range = np.array([timestep[0] - 4, timestep[0]])
                    box_p_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, :2] + np.array(
                        [x_min, y_min])  # (5, 2) global frame coord
                    box_h_gt = n.get(tr_scene=ts_range, state=eval_stg_nm.state['VEHICLE'])[:, 6:7]  # (5, 1)

                    box_p_gt = torch.tensor(box_p_gt).float().cuda()  # double -> float
                    box_h_gt = torch.tensor(box_h_gt).float().cuda()  # double -> float
                    break

            # clustering based on perturbations in current frame
            # load cluster
            db = DBSCAN(eps=cfg.CLUSTER.eps, min_samples=cfg.CLUSTER.min_samples)  # a smaller eps for more clusters
            db.fit(dh_set_det[:, attack_idx, :].detach().cpu().numpy())

            label_unique = np.unique(db.labels_)
            label_unique = label_unique[label_unique != -1]

            label_count = np.zeros(len(label_unique))
            for i, label in enumerate(label_unique):
                label_count[i] = np.sum(db.labels_ == label)

            # cluster stat based on current frame; single point attack
            # loop each cluster with sample num > 10
            cluster_perturb_curr = []
            cluster_perturb_stats = []
            for i, label in enumerate(label_unique):
                # get sample indexes in each cluster
                cluster_idxes = np.where(db.labels_ == label)[0]

                # get curr stat of each cluster (for perturbation init)
                dh_curr = dh_set_det[cluster_idxes, attack_idx, 0]
                dx_curr = dp_set_det[cluster_idxes, attack_idx, 0]
                dy_curr = dp_set_det[cluster_idxes, attack_idx, 1]

                # for initialization
                cluster_perturb_stats.append(np.array([
                    [torch.min(dh_curr).detach().cpu(), torch.max(dh_curr).detach().cpu(), torch.mean(dh_curr).detach().cpu(), torch.std(dh_curr).detach().cpu()],
                    [torch.min(dp_set_det[:, attack_idx, 0]).detach().cpu(), torch.max(dp_set_det[:, attack_idx, 0]).detach().cpu(), torch.mean(dp_set_det[:, attack_idx, 0]).detach().cpu(),
                     torch.std(dp_set_det[:, attack_idx, 0]).detach().cpu()],
                    [torch.min(dp_set_det[:, attack_idx, 1]).detach().cpu(), torch.max(dp_set_det[:, attack_idx, 1]).detach().cpu(), torch.mean(dp_set_det[:, attack_idx, 1]).detach().cpu(),
                     torch.std(dp_set_det[:, attack_idx, 1]).detach().cpu()]
                ]))

                # for matching loss
                cluster_perturb_curr.append({'dh': dh_curr, 'dx': dx_curr, 'dy': dy_curr})
            print('cluster num: {}'.format(len(cluster_perturb_curr)))
            ### step 1: init ###

            ### step 2: inference wo attack ###
            with torch.no_grad():
                # trajectron.py
                _, _, future_traces, predict_trajs_noatk = eval_stg_nm.run(scene,
                                                                           timestep,
                                                                           ph,
                                                                           min_future_timesteps=ph,
                                                                           num_samples=1,
                                                                           z_mode=True,
                                                                           gmm_mode=True,
                                                                           full_dist=False,
                                                                           all_z_sep=False)

                # egoplan_clean_scene = torch.tensor(plan(predict_trajs_noatk, scene, target_inst_token, ts_curr)).cuda()
                egoplan_clean_scene = torch.tensor(future_traces['ego'][:, :2]).cuda()
            ### step 2: inference wo attack ###

            ### step 3: pgd attack ###
            dh_set_pred = []
            dp_set_pred = []
            loss_set_pred = []

            for cluster_idx in range(len(cluster_perturb_stats)):
                for epoch in range(cfg.epoch_num):
                    # find min loss in each epoch
                    min_loss = 999.
                    min_dh = min_dp = None

                    ### epoch init ###
                    # re-init perturbation in each epoch
                    perturb_cand = {'obj_id': target_inst_token}
                    perturb_cand['duration'] = [-5, -4, -3, -2, -1]
                    perturb_cand['delta_heading'] = Variable(torch.zeros(5, 1)).cuda()
                    perturb_cand['delta_position'] = Variable(torch.zeros((5, 2))).cuda()

                    dh_min_curr, dh_max_curr, dh_mean_curr, dh_std_curr = \
                        cluster_perturb_stats[cluster_idx][0, 0], cluster_perturb_stats[cluster_idx][0, 1], \
                            cluster_perturb_stats[cluster_idx][0, 2], cluster_perturb_stats[cluster_idx][0, 3]
                    perturb_cand['delta_heading'][-1, 0] = Variable(torch.normal(torch.tensor(dh_mean_curr), torch.tensor(dh_std_curr)),
                                                                    requires_grad=True).cuda()

                    dx_min, dx_max, dx_mean, dx_std = \
                        cluster_perturb_stats[cluster_idx][1, 0], cluster_perturb_stats[cluster_idx][1, 1], \
                            cluster_perturb_stats[cluster_idx][1, 2], cluster_perturb_stats[cluster_idx][1, 3]
                    dy_min, dy_max, dy_mean, dy_std = \
                        cluster_perturb_stats[cluster_idx][2, 0], cluster_perturb_stats[cluster_idx][2, 1], \
                            cluster_perturb_stats[cluster_idx][2, 2], cluster_perturb_stats[cluster_idx][2, 3]
                    perturb_cand['delta_position'][-1, 0] = Variable(torch.normal(torch.tensor(dx_mean), torch.tensor(dx_std)),
                                                                     requires_grad=True).cuda()
                    perturb_cand['delta_position'][-1, 1] = Variable(torch.normal(torch.tensor(dy_mean), torch.tensor(dy_std)),
                                                                     requires_grad=True).cuda()

                    # create cluster perturb (*, 3)
                    cluster_perturb = torch.stack([cluster_perturb_curr[cluster_idx]['dh'], \
                                                   cluster_perturb_curr[cluster_idx]['dx'], \
                                                   cluster_perturb_curr[cluster_idx]['dy']], 0).transpose(0, 1)

                    # re-init learning rate in each epoch
                    dh_lr = cfg.dh_lr_init
                    dp_lr = cfg.dp_lr_init
                    ### epoch init ###

                    # start iteration
                    for iter in range(cfg.iter_num):

                        ### iter init ###
                        # update each iter after several steps
                        loss_steps = []
                        grad_dh_steps = []
                        grad_dp_steps = []

                        # create perturb (1, 3) for current iter
                        curr_perturb = torch.stack([perturb_cand['delta_heading'][-1, 0], \
                                                    perturb_cand['delta_position'][-1, 0], \
                                                    perturb_cand['delta_position'][-1, 1]], 0).unsqueeze(0)

                        # find top k cloest perturbation in cluster_perturb to curr_perturb using L1 distance
                        dist = torch.sum(torch.abs(cluster_perturb - curr_perturb), dim=1)
                        _, topk_idx = torch.topk(dist, k=1, largest=False, sorted=False)
                        cluster_perturb_closest = cluster_perturb[topk_idx, :]
                        ### iter init ###

                        # start steps
                        for step in range(cfg.step_num):
                            ### step init ###
                            target_dh_set = dh_set_det.reshape(-1, 1)  # [det_idx * frame_idx， 1]
                            target_dp_set = dp_set_det.reshape(-1, 2)  # [det_idx * frame_idx， 2]
                            ### step init ###

                            ### step attack ###
                            _, _, _, predict_trajs = eval_stg_nm.attack(scene,
                                                                        timestep,
                                                                        ph,
                                                                        min_future_timesteps=ph,
                                                                        num_samples=1,
                                                                        z_mode=True,
                                                                        gmm_mode=True,
                                                                        full_dist=False,
                                                                        all_z_sep=False,
                                                                        perturb_cand=perturb_cand,
                                                                        target_dh_set=target_dh_set,
                                                                        target_dp_set=target_dp_set)

                            ### step attack ###

                            ### calculate loss ###
                            ade_tar2ego = ade(predict_trajs[target_inst_token], egoplan_clean_scene)
                            l2_matching = loss_matching(curr_perturb, cluster_perturb_closest.unsqueeze(0))   # SP-ATTACK

                            # loss = ade_tar2ego
                            loss = ade_tar2ego + l2_matching
                            loss_steps.append(loss.item())
                            ### calculate loss ###

                            ### calc gradient ###
                            grad_dh = \
                            torch.autograd.grad(loss, perturb_cand['delta_heading'], retain_graph=True, create_graph=False,
                                                allow_unused=True)[0]
                            grad_dh_steps.append(grad_dh)
                            grad_dp = \
                            torch.autograd.grad(loss, perturb_cand['delta_position'], retain_graph=True, create_graph=False,
                                                allow_unused=True)[0]
                            grad_dp_steps.append(grad_dp)
                            ### calc gradient ###

                        ### reocrd best iter ###
                        loss_iter = abs(np.mean(loss_steps))
                        print('cluster {} epoch {} iter {}: loss {}'.format(cluster_idx, epoch, iter, round(loss_iter, 2)))

                        # record best perturbation
                        if iter > 0 and loss_iter <= min_loss:  # iteration 0 are not updated yet
                            min_loss = loss_iter
                            min_dh = perturb_cand['delta_heading'].clone()
                            min_dp = perturb_cand['delta_position'].clone()
                        ### reocrd best iter ###

                        ### update perturbation with average gradient ###
                        grad_dh = torch.mean(torch.stack(grad_dh_steps), dim=0)  # (5, 1)
                        grad_dp = torch.mean(torch.stack(grad_dp_steps), dim=0)  # (5, 2)

                        # update dh, dp on current frame only
                        perturb_cand['delta_heading'][-1].data -= dh_lr * torch.sign(grad_dh[-1])
                        perturb_cand['delta_position'][-1, 0].data -= dp_lr * torch.sign(grad_dp[-1, 0])
                        perturb_cand['delta_position'][-1, 1].data -= dp_lr * torch.sign(grad_dp[-1, 1])

                        # clamp perturbation
                        perturb_cand['delta_heading'][-1] = torch.clamp(perturb_cand['delta_heading'][-1], \
                                                                        min=dh_min_curr, max=dh_max_curr)
                        perturb_cand['delta_position'][-1, 0] = torch.clamp(perturb_cand['delta_position'][-1, 0], \
                                                                            min=dx_min, max=dx_max)
                        perturb_cand['delta_position'][-1, 1] = torch.clamp(perturb_cand['delta_position'][-1, 1], \
                                                                            min=dy_min, max=dy_max)

                        # adjust learning rate
                        dh_lr *= cfg.lr_decay
                        dp_lr *= cfg.lr_decay
                        ### update perturbation with average gradient ###

                    # record for each epoch
                    if (min_loss != 999.) and (min_dh != None) and (min_dp != None):
                        loss_set_pred.append(min_loss)
                        dh_set_pred.append(min_dh.detach().clone())
                        dp_set_pred.append(min_dp.detach().clone())

            ### step 3: pgd attack ###

            if len(dh_set_pred) == 0 or len(dp_set_pred) == 0:
                raise ValueError('{}: empty candidate set!'.format(scene_name))

            # save attack result
            heading_set_pred = [out_format(dh + box_h_gt) for dh in dh_set_pred]
            pos_set_pred = [out_format(dp + box_p_gt) for dp in dp_set_pred]

            dh_set_pred = [out_format(dh) for dh in dh_set_pred]
            dp_set_pred = [out_format(dp) for dp in dp_set_pred]

            top_k = 3
            top_k_idx = np.argsort(loss_set_pred)[:top_k]

            loss_set_pred = [loss_set_pred[i] for i in top_k_idx]
            dh_set_pred = [dh_set_pred[i][-1] for i in top_k_idx]
            dp_set_pred = [dp_set_pred[i][-1] for i in top_k_idx]
            heading_set_pred = [heading_set_pred[i][-1] for i in top_k_idx]
            pos_set_pred = [pos_set_pred[i][-1] for i in top_k_idx]

            inverse_res[scene_name + '_{}'.format(str(attack_idx+6))] = {
                'loss': loss_set_pred,
                'dh': dh_set_pred,
                'dp': dp_set_pred,
                'heading': heading_set_pred,
                'pos': pos_set_pred,
            }

        with open(os.path.join(curr_work_dir, 'inverse_res_multi_frame.json'), 'w') as f:
            json.dump(inverse_res, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnt/dataset/nuScenes/results')
    parser.add_argument('--scene_name', type=str, default='scene-0043')
    args = parser.parse_args()

    ### hyperparameters ###
    # constant
    ph = 6  # 6
    log_dir = './models'
    model_name = 'int_ee'  # 'int_ee_me'
    ts_node = 0
    ### hyperparameters ###

    ### load ###
    # load nuScenes data
    with open(os.path.join(args.data_dir, 'attack_pred/nuScenes_attack_{}.pkl'.format(args.scene_name)), 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')
    eval_scenes = eval_env.scenes

    # load model
    device = 'cuda:0'
    model_dir = os.path.join(log_dir, model_name)
    eval_stg_nm, hyp = load_model(model_dir, eval_env, ts=20, device=device)
    # TODO: need model_registrar-%d.pt

    ### load ###
    # generate heading and position perturbations, delta^{x, y, h}
    process()

    # launch multi-frame pgd attack
    attack()