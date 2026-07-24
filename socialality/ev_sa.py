"""
@Author: Ziqian Zou
@Date: 2026-01-22 09:48:21
@LastEditors: Ziqian Zou
@LastEditTime: 2026-07-24 09:45:21
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""

import torch

import qpid.mods.vis.helpers
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import SocialalityArgs
from ._groupingKernel import GroupingKernel
from ._perceptionMechanism import INF, PerceptionMechanism
from .egoLoss import EgoLoss
from .group_vis.groupVis import modify_qpid_utils


class EVSocialalityModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.sa_args = self.args.register_subargs(
            SocialalityArgs, 'sa_args')
        self.r = self.sa_args

        # Set model inputs
        inputs = [INPUT_TYPES.OBSERVED_TRAJ,
                  INPUT_TYPES.NEIGHBOR_TRAJ]
        if self.r.use_team_group_mask:
            inputs.append('TEAM_GROUP_MASK')
        self.set_inputs(*inputs)

        # Layers
        tlayer, itlayer = layers.get_transform_layers('haar')

        # Transform layers
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((len(self.output_pred_steps), self.dim))

        # Bilinear structure (outer product + pooling + fc)
        # For trajectories
        self.outer = layers.OuterLayer(self.d//2, self.d//2)
        self.pooling = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = layers.Dense((self.d//4)**2, self.d//2, torch.nn.Tanh)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.dim, self.d//2,
                                      torch.nn.Tanh,
                                      transform_layer=self.t1)

        # Grouping kernel
        self.grouping = GroupingKernel(
            traj_dim=self.dim,
            feature_dim=self.r.output_units,
            obs_steps=self.args.obs_frames,
            pred_steps=self.args.pred_frames,
            insights=self.r.insights_num,
            backbone=self.r.ego_predictor_type,
            ego_capacity=self.r.ego_capacity,
            ego_t_f=self.r.ego_t_f,
            ego_t_h=self.r.ego_t_h,
            use_mixed=self.r.use_mixed_trajectory,
            fix_dis_anchor = self.r.fix_distance_anchor,
            fix_speed_anchor = self.r.fix_speed_anchor,
            set_anchor = self.r.set_anchor_value,
            set_dis_anchor = self.r.set_distance_anchor,
            set_speed_anchor = self.r.set_speed_anchor,
            previews_only = self.r.previews_only,
            vis_anchors = self.r.vis_anchors,
            disable_dis_anchor = self.r.disable_distance_anchor,
            disable_speed_anchor = self.r.disable_speed_anchor,
            current_only = self.r.current_only,
            set_grouping_ratio = self.r.set_grouping_ratio,
        )

        # Perception mechanism
        self.perception = PerceptionMechanism(
            traj_dim=self.dim,
            feature_dim=self.r.output_units,
            view_angle=self.r.view_angle,
            adaptive_fov = self.r.adaptive_fov,
            max_agents = self.args.max_agents,
        )

        # Concat all ego, group, out-of-group agents feature and encode
        self.concat_fc = layers.Dense(
            self.r.output_units * 5,
            self.r.output_units * 4,
            activation=torch.nn.Tanh
        )

        # Linear prediction of obs as the target of transformer
        self.lp = layers.LinearLayerND(
            self.args.obs_frames,
            self.args.pred_frames,
            return_full_trajectory=True
        )

        # Backbone
        self.bb = transformer.Transformer(
            num_layers=4,
            d_model=self.r.output_units * 4,
            num_heads=8,
            dff=512,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.pred_frames + self.args.obs_frames,
            include_top=False
        )

        self.T = transformer.Transformer(
            num_layers=4,
            d_model=192,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Noise encoding
        self.ie = torch.nn.Sequential(
            layers.Dense(input_units=self.d_id, 
                         output_units=self.r.output_units * 4, 
                         activation=torch.nn.ReLU),
            layers.Dense(input_units=self.r.output_units * 4, 
                         output_units=self.r.output_units * 4, 
                         activation=torch.nn.Tanh),
        )

        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(self.r.output_units * 8,
                                  self.r.generation_num,
                                  torch.nn.Tanh)
        self.ms_fc_t = layers.Dense(self.r.output_units * 6,
                                  self.r.generation_num,
                                  torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.r.output_units * 4, self.r.output_units * 8)
        self.ms_conv_t = layers.GraphConv(self.r.output_units * 6, self.r.output_units * 8)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.r.output_units * 8, self.r.output_units * 8, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.r.output_units * 8,
                                        self.args.pred_frames * self.dim)
        
        self.decoder_fc1_t = layers.Dense(self.r.output_units * 8, self.r.output_units * 8, torch.nn.Tanh)
        self.decoder_fc2_t = layers.Dense(self.r.output_units * 8,
                                        self.args.pred_frames * self.dim)
        
    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Trajectory embedding and encoding
        f_ego_t = self.te(x_ego)
        f_ego_t = self.outer(f_ego_t, f_ego_t)
        f_ego_t = self.pooling(f_ego_t)
        f_ego_t = self.flatten(f_ego_t)
        f_ego_t = self.outer_fc(f_ego_t)       # (batch, steps, 64)

        group_mask, trajs_group, f_ego, socialality, nei_pred_train, y_nei, grouping_justifications = self.grouping(
            x_ego, 
            x_nei, 
            training)
        
        # Trajectory embedding and encoding
        f_group_t = self.te(trajs_group)
        f_group_t = self.outer(f_group_t, f_group_t)
        f_group_t = self.pooling(f_group_t)
        f_group_t = self.flatten(f_group_t)
        f_group_t = self.outer_fc(f_group_t)       # (batch, max_agents, steps, 64)
        f_group_t_idx = torch.max(torch.sum(f_group_t**2, dim=[-1, -2]), dim=1)
        batch_idx_group = torch.arange(f_group_t.shape[0], device=f_group_t.device)
        f_group_t = f_group_t[batch_idx_group, f_group_t_idx.indices]

        if self.r.use_team_group_mask:
            group_mask = self.get_input(inputs, 'TEAM_GROUP_MASK')
            trajs_group = (grouping_justifications * group_mask[..., None, None]).to(dtype=torch.float32)

        # ----------------------------
        # MARK: - Perception Mechanism
        # ----------------------------
        f_group, f_out_group = self.perception(
            x_ego, 
            x_nei, 
            group_mask, 
            trajs_group)
        
        # Mask neighbors
        nei_mask = torch.sum(x_nei.abs(), dim=[-1, -2]) < (0.05 * INF)
        nei_mask = nei_mask.to(dtype=torch.int32)

        out_group_mask = (1 - group_mask) * nei_mask
        trajs_out_group  = (
            x_nei * out_group_mask[..., None, None]).to(dtype=torch.float32)
        
        # Trajectory embedding and encoding
        f_out_group_t = self.te(trajs_out_group)
        f_out_group_t = self.outer(f_out_group_t, f_out_group_t)
        f_out_group_t = self.pooling(f_out_group_t)
        f_out_group_t = self.flatten(f_out_group_t)
        f_out_group_t = self.outer_fc(f_out_group_t)       # (batch, max_agents, steps, 64)
        f_out_group_t_idx = torch.max(torch.sum(f_out_group_t**2, dim=[-1, -2]), dim=1)
        batch_idx_out_group = torch.arange(f_out_group_t.shape[0], device=f_out_group_t.device)
        f_out_group_t = f_out_group_t[batch_idx_out_group, f_out_group_t_idx.indices]

        f_t = f = torch.concat([f_ego_t, f_group_t, f_out_group_t], dim=-1)


        # -----------------------
        # MARK: - Fusion Strategy
        # -----------------------
        # ablation args `disable_distance_anchor` and `disable_speed_anchor`
        # are also used here
        if not self.r.disable_distance_anchor and not self.r.disable_speed_anchor:
            if not self.r.remove_modulation:
                f_ego = f_ego * (1.0 + socialality[..., None, -1:])
                f_group = f_group * (1.0 / (1.0 + socialality[..., None, :1]))
                f_out_group = (f_out_group *
                            (1 / (1 + socialality[..., None, -1:])) *
                            (1 / (1 + socialality[..., None, :1]))) 
        
        # fusion strategy when disable distance anchor
        if self.r.disable_distance_anchor:
            f_ego = f_ego * (1.0 + socialality[..., None, -1:])
            f_group = f_group
            f_out_group = (f_out_group *
                        (1 / (1 + socialality[..., None, -1:])))
        
        # fusion strategy when disable speed anchor
        if self.r.disable_speed_anchor:
            f_ego = f_ego
            f_group = f_group * (1.0 / (1.0 + socialality[..., None, :1]))
            f_out_group = (f_out_group *
                        (1 / (1 + socialality[..., None, :1])))

        f = torch.concat([f_ego, f_group, f_out_group], dim=-1)
        f = self.concat_fc(f)

        # ------------------------------------
        # MARK: - Backbone (Transformer & MSN)
        # ------------------------------------
        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        pred_linear = self.lp(x_ego)
        pred_linear = (pred_linear - 
                       pred_linear[..., None, self.args.obs_frames - 1, :])

        # (batch, obs + pred, out_uni * 4)
        f_tran, _ = self.bb(inputs=f, targets=pred_linear, training=training)

        traj_targets = self.t1(x_ego)
        f_tran_t, _ = self.T(inputs=f_t, targets=traj_targets, training=training)

        # Multiple generations -> (batch, Kc, d)
        adj_t = self.ms_fc_t(f_t)               # (batch, steps, Kc)
        adj_t = torch.transpose(adj_t, -1, -2)
        f_multi_t = self.ms_conv_t(f_tran_t, adj_t)     # (batch, Kc, d)

        # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
        y_t = self.decoder_fc1_t(f_multi_t)
        y_t = self.decoder_fc2_t(y_t)
        y_t = torch.reshape(y_t, list(y_t.shape[:-1]) +
                            [self.Tsteps_de, self.Tchannels_de])

        y_t = self.it1(y_t)

        # Slice the prediction time steps
        # (batch, pred, out_uni * 4)
        f_tran = f_tran[:, self.args.obs_frames:, ...]

        # Prediction
        for _ in range(repeats):
            # Assign random ids and embedding
            z = torch.normal(mean=0, std=1, size=list(
                f_tran.shape[:-1]) + [self.d_id])
            # (batch, pred, out_uni * 4)
            f_z = self.ie(z.to(f_tran.device))

            # (batch, pred, out_uni * 8)
            f_final = torch.concat([f_tran, f_z], dim=-1)

            # Multiple generations -> (batch, Kc, out_uni * 8)
            # (batch, steps, Kc)
            adj = self.ms_fc(f_final)
            adj = torch.transpose(adj, -1, -2)
            # (batch, Kc, out_uni * 4)
            f_multi = self.ms_conv(f_tran, adj)

            # decode trajectories
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y,
                              list(y.shape[:-1]) + [self.args.pred_frames,
                                                    self.dim])
            
            y = y + y_t

            all_predictions.append(y)

        Y = torch.concat(all_predictions, dim=-3)

        returns = [
            Y + pred_linear[..., None, self.args.obs_frames:, :],
        ]

        # Output predictions and labels to compute EgoLoss
        if training:
            returns += [
                x_nei[..., -self.r.ego_t_f:, :],
                nei_pred_train,
            ]

        # ---------------------
        # MARK: - Visualization
        # ---------------------
        # Visualize ego predictor's outputs
        # This only works in the playground mode
        elif v := self.r.vis_ego_predictor:
            match v:
                case 1:
                    e = y_nei
                case 2:
                    e = y_nei
                case _:
                    self.log(f'Wrong `vis_ego_predictor` value recevied: {v}!',
                             level='error', raiseError=ValueError)

            returns[0] = e
        
        if self.r.vis_group_members:
            # group member visualization
            returns[0] = torch.flatten(trajs_group[..., 
                                                   self.r.ego_t_h-1:self.r.ego_t_h, :],
                                                     -3, -2)
            
            if self.r.vis_grouping_window:
            # grouping stage visualization
                returns[0] = torch.flatten(grouping_justifications[:, torch.where(group_mask[0])[0]],
                                                     -3, -2)
        else:
            if self.r.vis_grouping_window:
                self.log('Arg `vis_grouping_window` can be only used ' + 
                         'when arg `vis_group_members` is activated!',
                     level='error', raiseError=ValueError)
    
        return returns
        

class EVSocialality(Structure):
    MODEL_TYPE = EVSocialalityModel

    def __init__(self, args=None,
                 manager=None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        self.r = self.args.register_subargs(
            SocialalityArgs, 'sa_args')

        if (r := self.r.ego_capacity) > (m := self.args.max_agents):
            self.log(f'Wrong capacity settings: {r} > {m}!',
                     level='error', raiseError=ValueError)

        if ((self.r.group_type == 1)
                and (self.r.ego_predictor_type != 'linear')):
            self.loss.set({l2: self.r.l2_loss_ratio,
                           EgoLoss: self.r.ego_loss_ratio})
        else:
            self.loss.set({l2: 1.0})
        
        modify_qpid_utils(mod_pred_img=self.r.vis_group_members, 
                          mod_vis_func=self.r.vis_ego_predictor + self.r.vis_group_members,
                          mod_vis_type=self.r.vis_group_members + self.r.vis_grouping_window)
        
