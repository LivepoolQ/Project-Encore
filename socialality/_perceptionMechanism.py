"""
@Author: Ziqian Zou
@Date: 2026-01-22 11:00:02
@LastEditors: Ziqian Zou
@LastEditTime: 2026-01-22 14:24:25
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2026 Ziqian Zou, All Rights Reserved.
"""
import numpy as np
import torch

from qpid.model import layers

from .utils import repeat

INF = 100000000
MU = 0.00001


class PerceptionMechanism(torch.nn.Module):
    def __init__(self, 
                 *args, 
                 traj_dim: int,
                 feature_dim: int,
                 view_angle: float = np.pi,
                 **kwargs):
        super().__init__()

        self.traj_dim = traj_dim
        self.feature_dim = feature_dim
        self.view_angle = view_angle

        # Group trajectory encoding
        self.ge = torch.nn.Sequential(
            layers.Dense(input_units=self.traj_dim, 
                         output_units=self.feature_dim, 
                         activation=torch.nn.ReLU),
            layers.Dense(input_units=self.feature_dim, 
                         output_units=self.feature_dim, 
                         activation=torch.nn.Tanh),
        )

        # Out-group perception
        self.outper = HumanPerception(
            feature_dim=self.feature_dim,
            view_angle=self.view_angle,
        )

    def forward(self, 
                x_ego: torch.Tensor,
                x_nei: torch.Tensor,
                group_mask: torch.Tensor, 
                trajs_group: torch.Tensor):
        # --------------------
        # MARK: - In-group perception
        # --------------------
        f_group = self.ge(trajs_group) * group_mask[..., None, None]
        f_group = torch.max(f_group, dim=-3)[0]
        
        # --------------------
        # MARK: - Out-group perception
        # --------------------
        # Mask neighbors
        nei_mask = torch.sum(x_nei.abs(), dim=[-1, -2]) < (0.05 * INF)
        nei_mask = nei_mask.to(dtype=torch.int32)
        
        # Apply mask on neighbors
        x_nei = ((1 - group_mask[..., None, None]) * x_nei +
                     group_mask[..., None, None] * INF)
        
        # Calculate out-of-group neighbors' mask
        out_group_mask = nei_mask * (1 - group_mask)
        x_nei = x_nei * out_group_mask[..., None, None]

        # Out-group perception
        f_out_group = self.outper(x_ego, x_nei)
        f_out_group = repeat(f_out_group[..., None, :], x_ego.shape[-2], dim=-2)

        return f_group, f_out_group
    
    
class HumanPerception(torch.nn.Module):

    def __init__(self, 
                 feature_dim: int,
                 view_angle: float = np.pi,
                 *args, **kwargs):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.view_angle = view_angle

        self.region_emb = torch.nn.Sequential(
            layers.Dense(3, self.feature_dim, torch.nn.ReLU),
            layers.Dense(self.feature_dim, self.feature_dim, torch.nn.ReLU),
            layers.Dense(self.feature_dim, self.feature_dim, torch.nn.Tanh)
        )

        self.concat_fc = torch.nn.Sequential(
            layers.Dense(3, self.feature_dim, torch.nn.ReLU),
            layers.Dense(self.feature_dim, self.feature_dim * 2, torch.nn.ReLU),
            layers.Dense(self.feature_dim * 2, self.feature_dim, torch.nn.Tanh)
        )

    def forward(self, trajs: torch.Tensor, nei_trajs: torch.Tensor):

        # `nei_trajs` are relative values to target agents' last obs step
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :] - trajs[..., -1:, :]

        # obs's direction is simplified to be its moving direction during the last interval
        obs_dir_vec = trajs[..., -1:, :] - trajs[..., -2:-1, :]
        obs_dir = torch.atan2(obs_dir_vec[..., 0], obs_dir_vec[..., 1])
        obs_dir = obs_dir % (2*np.pi)

        # neighbor's direction
        nei_dir = torch.atan2(nei_posion_vector[..., 0],
                              nei_posion_vector[..., 1])
        nei_dir = nei_dir % (2*np.pi)

        # mask neighbors
        nei_mask = (
            torch.sum(torch.abs(nei_trajs), dim=[-1, -2]) > 0).to(dtype=torch.int32)

        # mask view angle
        view_mask = (torch.abs(nei_dir - obs_dir) <
                     (self.view_angle / 2)) * nei_mask.to(dtype=torch.int32)
        left_view_mask = view_mask * ((nei_dir - obs_dir) > 0).to(dtype=torch.int32)
        right_view_mask = view_mask - left_view_mask

        # mask back angle(places out of the view)
        back_mask = nei_mask - view_mask

        # all real neighbors in left view, right view and back view
        nei_left = left_view_mask * nei_mask
        nei_right = right_view_mask * nei_mask
        nei_back = back_mask * nei_mask

        # region encoding
        region_ids = torch.eye(3).to(trajs.device)
        region_vec = self.region_emb(region_ids)

        # calculate neighbors' distance
        dis = torch.norm(nei_posion_vector, dim=-1)

        # calculate neighbors' moving direction
        nei_move_dir_vec = nei_trajs[..., -1:, :] - nei_trajs[..., -2:-1, :]
        nei_move_dir = torch.atan2(
            nei_move_dir_vec[..., 0] + MU, nei_move_dir_vec[..., 1] + MU)
        nei_move_dir = nei_move_dir % (2*np.pi)
        delta_dir = torch.squeeze(
            (nei_move_dir - obs_dir[:, None, ...]), dim=-1)

        # calculate neighbor's velocity
        velocity = torch.norm(nei_vector, dim=-1)

        # for neighbors in view angle, the conception layer would consider all three factors
        # for neighbors in the back, the conception layer would only consider distance factor
        # calculate conception value in right view
        dis_right = (torch.sum(dis * nei_right,
                               dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
        dir_right = (torch.sum(delta_dir * nei_right,
                               dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
        vel_right = (torch.sum(velocity * nei_right,
                               dim=[-1, -2])) / (torch.sum(nei_right, dim=-1) + MU)
        con_right = torch.concat(
            [dis_right[:, None, None], dir_right[:, None, None], vel_right[:, None, None]], dim=-1)

        # calculate conception value in left view
        dis_left = (torch.sum(dis * nei_left,
                    dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
        dir_left = (torch.sum(delta_dir * nei_left,
                    dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
        vel_left = (torch.sum(velocity * nei_left,
                    dim=[-1, -2])) / (torch.sum(nei_left, dim=-1) + MU)
        con_left = torch.concat(
            [dis_left[:, None, None], dir_left[:, None, None], vel_left[:, None, None]], dim=-1)

        # calculate conception in the back
        dis_back = (torch.sum(dis * nei_back,
                    dim=[-1, -2])) / (torch.sum(nei_back, dim=-1) + MU)
        dir_back = torch.zeros_like(dir_left)
        vel_back = torch.zeros_like(vel_left)
        con_back = torch.concat([dis_back[:, None, None], dir_back[:, None, None], vel_back[:, None, None]], dim=-1)

        f = torch.concat([con_right, con_left, con_back], dim=-2)
        f = self.concat_fc(f)
        f = f + region_vec[None]
        
        return f.reshape(f.shape[0], -1)