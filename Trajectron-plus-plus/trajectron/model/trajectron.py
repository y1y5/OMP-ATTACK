import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore
from environment.data_utils import derivative_of


def sample_ego_states(ego_state, vel_scale=1.0, dt=0.5):

    # Initialize the state of the vehicle
    ego_state_sim = np.zeros_like(ego_state) # shape is (5, 8)

    # Set the state at frame t
    ego_state_sim[-1] = ego_state[-1]

    # Compute the state at frames t-1, t-2, t-3, t-4 using the equations of motion in reverse
    for t in range(4, 0, -1):
        # Scale the velocity
        scaled_velocity = ego_state[t-1, 2:4] * vel_scale

        # Compute the acceleration based on the change in velocity
        acceleration = (scaled_velocity - ego_state_sim[t, 2:4]) / dt

        # Update the acceleration and velocity for the previous frame
        ego_state_sim[t-1, 4:6] = acceleration
        ego_state_sim[t-1, 2:4] = scaled_velocity
        ego_state_sim[t-1, 0:2] = ego_state_sim[t, 0:2] - scaled_velocity*dt

    # change delta heading accordingly
    # Note: derivative should only apply non nan frames, or the output will be all 0.
    ego_state_sim[:,-1] = derivative_of(ego_state_sim[:, 6], dt, radian=True)

    return ego_state_sim


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        # {'position': ['x', 'y'], 'velocity': ['x', 'y'], 'acceleration': ['x', 'y'], 'heading': ['°', 'd°']}
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        # {"VEHICLE": {"position": ["x", "y"]}, "PEDESTRIAN": {"position": ["x", "y"]}}
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict
    
    # original inference
    def run(self,
            scene,
            timesteps,
            ph,
            num_samples=1,
            min_future_timesteps=0,
            min_history_timesteps=1,
            z_mode=False,
            gmm_mode=False,
            full_dist=True,
            all_z_sep=False):

        predictions_dict = {}
        observe_traces = {}
        future_traces = {}
        predict_traces = {}

        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            y = y_t.to(self.device)

            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            for i, n in enumerate(nodes):
                # if n.id not in [perturbation["obj_id"], 'ego']:
                #     continue
                observe_traces[n.id] = x[i][:,[0, 1, -2, -1]]  # position: ['x', 'y'], heading: ['°', 'd°']
                future_traces[n.id] = y[i][:,:2]
                predict_traces[n.id] = predictions[0][i][:,:2]

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict, observe_traces, future_traces, predict_traces

    # inference with det
    def eval(self,
            scene,
            timesteps,
            ph,
            num_samples=1,
            min_future_timesteps=0,
            min_history_timesteps=1,
            z_mode=False,
            gmm_mode=False,
            full_dist=True,
            all_z_sep=False,
            det_cand=None,
            ego_pose_sample=None,
            dt = 1/2):

        predictions_dict = {}
        observe_traces = {}
        future_traces = {}
        predict_traces = {}

        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]
                
            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            
            ### change ego node ###
            if ego_pose_sample is not None:
                # get ego node
                for i, n in enumerate(nodes):
                    if n.id == 'ego':
                        ego_idx = i
                        break

                ego_position = ego_pose_sample[:, :2]
                ego_heading = ego_pose_sample[:, 2:3] 
                
                # change position
                x[ego_idx][-5:, :2] = ego_position
                x_st_t[ego_idx][-5:, :2] = x[ego_idx][-5:, :2] / 80  # standardize (find dev in process_data.py)

                # change vx, vy accordingly
                x[ego_idx][-5:, 2:4] = torch.cat((torch.reshape(x[ego_idx][-4, 0:2]-x[ego_idx][-5, 0:2], (1,2)), x[ego_idx][-4:, 0:2]-x[ego_idx][-5:-1, 0:2]), 0) / dt
                x_st_t[ego_idx][-5:, 2:4] = (x[ego_idx][-5:, 2:4] - 0) / 15  # standardize

                # change ax, ay accordingly
                x[ego_idx][-5:, 4:6] = torch.cat((torch.reshape(x[ego_idx][-4, 2:4]-x[ego_idx][-5, 2:4], (1,2)), x[ego_idx][-4:, 2:4]-x[ego_idx][-5:-1, 2:4]), 0) / dt
                x_st_t[ego_idx][-5:, 4:6] = (x[ego_idx][-5:, 4:6] - 0) / 4  # standardize   

                # change heading (range [-pi, pi])
                x[ego_idx][-5:,-2:-1] = ego_heading
                x[ego_idx][-5:,-2:-1] = (x[ego_idx][-5:,-2:-1] + np.pi) % (2*np.pi) - np.pi  # normalize to [-pi, pi]
                x_st_t[ego_idx][-5:,-2:-1] = (x[ego_idx][-5:,-2:-1] - 0) / np.pi  # standardize to [-1, 1]
                                
                # change delta heading accordingly
                # Note: derivative should only apply non nan frames, or the output will be all 0.
                x[ego_idx][-5:,-1] = torch.tensor(derivative_of(x[ego_idx][-5:,-2].detach().cpu().numpy(), dt, radian=True), device=self.device)
                x_st_t[ego_idx][-5:,-1] = x[ego_idx][-5:,-1] / 1.  # standardize
                # print(x[ego_idx][-5:,:])
                # raise ValueError()
            ### change ego node ###

            #### change target node: assign det ###
            if det_cand is not None:
                
                # find index of target node
                target_index = -1
                for i, n in enumerate(nodes):
                    if n.id == det_cand["obj_id"]:
                        target_index = i
                        break

                # print(x[target_index][:,:2])

                duration = det_cand["duration"]
                pos = det_cand["delta_position"]
                heading = det_cand["delta_heading"]

                # change position
                x[target_index][duration,:2] = pos[duration]
                x_st_t[target_index][-5:,:2] = x[target_index][-5:,:2] / 80  # standardize (find dev in process_data.py)

                # change vx, vy accordingly
                x[target_index][-5:, 2:4] = torch.cat((torch.reshape(x[target_index][-4, 0:2]-x[target_index][-5, 0:2], (1,2)), x[target_index][-4:, 0:2]-x[target_index][-5:-1, 0:2]), 0) / dt
                x_st_t[target_index][-5:, 2:4] = (x[target_index][-5:, 2:4] - 0) / 15  # standardize

                # change ax, ay accordingly
                x[target_index][-5:, 4:6] = torch.cat((torch.reshape(x[target_index][-4, 2:4]-x[target_index][-5, 2:4], (1,2)), x[target_index][-4:, 2:4]-x[target_index][-5:-1, 2:4]), 0) / dt
                x_st_t[target_index][-5:, 4:6] = (x[target_index][-5:, 4:6] - 0) / 4  # standardize

                # change heading (range [-pi, pi])
                x[target_index][duration,-2:-1] = heading[duration]
                x_st_t[target_index][-5:,-2:-1] = (x[target_index][-5:,-2:-1] - 0) / np.pi  # standardize to [-1, 1]
                                
                # change delta heading accordingly
                # Note: derivative should only apply non nan frames, or the output will be all 0.
                x[target_index][-5:,-1] = torch.tensor(derivative_of(x[target_index][-5:,-2].detach().cpu().numpy(), dt, radian=True), device=self.device)
                x_st_t[target_index][-5:,-1] = x[target_index][-5:,-1] / 1.  # standardize
            #### change target node: assign det ###

            y = y_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,  # None
                                        map=map,  # None
                                        prediction_horizon=ph,  # 6
                                        num_samples=num_samples,  # 1
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            for i, n in enumerate(nodes):
                # if n.id not in [perturbation["obj_id"], 'ego']:
                #     continue
                observe_traces[n.id] = x[i][:,[0, 1, -2, -1]]  # position: ['x', 'y'], heading: ['°', 'd°']
                future_traces[n.id] = y[i][:,:2]
                predict_traces[n.id] = predictions[0][i][:,:2]

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict, observe_traces, future_traces, predict_traces

    # inference with perturbation
    def attack(self,
            scene,
            timesteps,
            ph,
            num_samples=1,
            min_future_timesteps=0,
            min_history_timesteps=1,
            z_mode=False,
            gmm_mode=False,
            full_dist=True,
            all_z_sep=False,
            perturb_cand=None,
            ego_vel_scale=None,
            target_dh_set=None,
            target_dp_set=None,
            dt = 1/2):

        predictions_dict = {}
        observe_traces = {}
        future_traces = {}
        predict_traces = {}

        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # print(timesteps, min_future_timesteps, min_history_timesteps)  # [4], 6, 1
                
            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)


            ### change ego node ###
            if ego_vel_scale is not None:
                # get ego node
                for i, n in enumerate(nodes):
                    if n.id == 'ego':
                        ego_idx = i
                        break
                
                ego_states = x[ego_idx][-5:,:]  # torch size (5, 8)
                # print(x[ego_idx][-5:,:])
                ego_states_sampled = sample_ego_states(ego_states.detach().cpu().numpy(), vel_scale=ego_vel_scale)  # numpy size (1, 5, 8)
                x[ego_idx][-5:,:] = torch.from_numpy(ego_states_sampled).to(self.device)  # torch size (5, 8)
            ### change ego node ###

            #### change target node ###
            # print(x[target_index][-5:,:2] + torch.tensor([scene.patch[0], scene.patch[1]]).cuda())
            # print(x[target_index][-5:,-2:-1])
            if perturb_cand is not None:
                
                target_index = -1
                for i, n in enumerate(nodes):
                    if n.id == perturb_cand["obj_id"]:
                        target_index = i
                        break

                duration = perturb_cand["duration"]
                dh = perturb_cand["delta_heading"].clone()
                dp = perturb_cand["delta_position"].clone()
                
                # random sample detection distribution, assign to first 4 frames
                if target_dh_set is not None:
                    random_indices = torch.randperm(target_dh_set.shape[0])[:4]
                    random_samples = target_dh_set[random_indices]
                    dh[:-1, :] = random_samples

                # random sample detection distribution, assign to first 4 frames
                if target_dp_set is not None:
                    # calc stat for dp
                    dx_mean, dx_std = torch.mean(target_dp_set[:, 0]), torch.std(target_dp_set[:, 0])
                    dy_mean, dy_std = torch.mean(target_dp_set[:, 1]), torch.std(target_dp_set[:, 1])

                    # assume dx, dy are normal distribution, sample from it
                    dp[:-1, 0] = torch.normal(dx_mean, dx_std, size=(4,))
                    dp[:-1, 1] = torch.normal(dy_mean, dy_std, size=(4,))
                
                # perturb position
                x[target_index][duration,:2] += dp[duration]
                x_st_t[target_index][-5:,:2] = x[target_index][duration,:2] / 80  # standardize (find dev in process_data.py)

                # change vx, vy accordingly
                x[target_index][-5:, 2:4] = torch.cat((torch.reshape(x[target_index][-4, 0:2]-x[target_index][-5, 0:2], (1,2)), x[target_index][-4:, 0:2]-x[target_index][-5:-1, 0:2]), 0) / dt
                x_st_t[target_index][-5:, 2:4] = (x[target_index][-5:, 2:4] - 0) / 15  # standardize

                # change ax, ay accordingly
                x[target_index][-5:, 4:6] = torch.cat((torch.reshape(x[target_index][-4, 2:4]-x[target_index][-5, 2:4], (1,2)), x[target_index][-4:, 2:4]-x[target_index][-5:-1, 2:4]), 0) / dt
                x_st_t[target_index][-5:, 4:6] = (x[target_index][-5:, 4:6] - 0) / 4  # standardize

                x[target_index][duration,-2:-1] += dh[duration]
                x[target_index][-5:,-2:-1] = (x[target_index][-5:,-2:-1] + np.pi) % (2*np.pi) - np.pi  # normalize to [-pi, pi]
                x_st_t[target_index][-5:,-2:-1] = (x[target_index][-5:,-2:-1] - 0) / np.pi  # standardize to [-1, 1]
                                
                # change delta heading accordingly
                # Note: derivative should only apply non nan frames, or the output will be all 0.
                x[target_index][-5:,-1] = torch.tensor(derivative_of(x[target_index][-5:,-2].detach().cpu().numpy(), dt, radian=True), device=self.device)
                x_st_t[target_index][-5:,-1] = x[target_index][-5:,-1] / 1.  # standardize
                #### add perturbation to target node ###

            y = y_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            for i, n in enumerate(nodes):
                # if n.id not in [perturbation["obj_id"], 'ego']:
                #     continue
                observe_traces[n.id] = x[i][:,[0, 1, -2, -1]]  # position: ['x', 'y'], heading: ['°', 'd°']
                future_traces[n.id] = y[i][:,:2]
                predict_traces[n.id] = predictions[0][i][:,:2]

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict, observe_traces, future_traces, predict_traces

