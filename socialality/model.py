"""
@Author: Ziqian Zou
@Date: 2026-01-22 09:48:21
@LastEditors: Ziqian Zou
@LastEditTime: 2026-02-06 11:23:39
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import SocialalityArgs
from ._groupingKernel import GroupingKernel
from ._perceptionMechanism import PerceptionMechanism
from .egoLoss import EgoLoss
import qpid.mods.vis.helpers
from .group_vis.groupVis import modify_qpid_utils


class SocialalityModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.sa_args = self.args.register_subargs(
            SocialalityArgs, 'sa_args')
        self.r = self.sa_args

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

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
        )

        # Perception mechanism
        self.perception = PerceptionMechanism(
            traj_dim=self.dim,
            feature_dim=self.r.output_units,
            view_angle=self.r.view_angle,
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
        self.ms_conv = layers.GraphConv(self.r.output_units * 4, self.r.output_units * 8)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.r.output_units * 8, self.r.output_units * 8, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.r.output_units * 8,
                                        self.args.pred_frames * self.dim)
        
    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # -----------------------
        # MARK: - Grouping Kernel
        # -----------------------
        group_mask, trajs_group, f_ego, socialality, nei_pred_train, y_nei = self.grouping(
            x_ego, 
            x_nei, 
            training)

        # ----------------------------
        # MARK: - Perception Mechanism
        # ----------------------------
        f_group, f_out_group = self.perception(
            x_ego, 
            x_nei, 
            group_mask, 
            trajs_group)

        # -----------------------
        # MARK: - Fusion Strategy
        # -----------------------
        f_ego = f_ego * (1.0 + socialality[..., None, -1:])
        f_group = f_group * (1.0 / (1.0 + socialality[..., None, :1]))
        f_out_group = (f_out_group *
                       (1 / (1 + socialality[..., None, -1:])) *
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
            if self.r.use_mixed_trajectory != 1:
                returns[0] = torch.flatten(trajs_group[..., 
                                                   -1:, :],
                                                     -3, -2)
            elif self.r.previews_only:
                returns[0] = torch.flatten((group_mask[..., None, None] * x_nei)[..., 
                                                   -1:, :],
                                                     -3, -2)
            elif self.r.use_mixed_trajectory == 1:
                returns[0] = torch.flatten(trajs_group[..., 
                                                   self.r.ego_t_h-1:self.r.ego_t_h, :],
                                                     -3, -2)
        
        return returns
        

class Socialality(Structure):
    MODEL_TYPE = SocialalityModel

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
        
        modify_qpid_utils(mod_pred_img=self.r.vis_group_members, mod_vis_func=self.r.vis_ego_predictor + self.r.vis_group_members)
        
