"""
@Author: Ziqian Zou
@Date: 2025-11-25 10:28:07
@LastEditors: Ziqian Zou
@LastEditTime: 2026-01-08 16:01:38
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2025 Ziqian Zou, All Rights Reserved.
"""
import numpy as np
import torch

from qpid.model import layers

from .__args import RangerArgs

INF = 100000000
MU = 0.00001


class GroupLayer(torch.nn.Module):

    def __init__(self,
                 output_units: int,
                 view_angle: float = np.pi,
                 *args, **kwargs):
        super().__init__()
        self.view_angle = view_angle
        self.output_units = output_units

        self.region_emb = torch.nn.Sequential(
            layers.Dense(3, output_units, torch.nn.ReLU),
            layers.Dense(output_units, output_units, torch.nn.ReLU),
            layers.Dense(output_units, output_units, torch.nn.Tanh)
        )

        self.concat_fc = torch.nn.Sequential(
            layers.Dense(3, output_units, torch.nn.ReLU),
            layers.Dense(output_units, output_units * 2, torch.nn.ReLU),
            layers.Dense(output_units * 2, output_units, torch.nn.Tanh)
        )

    def forward(self, trajs: torch.Tensor, nei_trajs: torch.Tensor, reshape=True):
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
        
        if reshape:
            return f.reshape(f.shape[0], -1)
        else:
            return f


class LongTermKernel(torch.nn.Module):

    def __init__(self, group_distance: int,
                 obs_steps: int,
                 *args, **kwargs):
        super().__init__()

        self.group_distance = 1.0
        self.obs_steps = obs_steps

    def forward(self, x_ego_2d: torch.Tensor, x_nei_2d: torch.Tensor, tolerance: torch.Tensor):

        ego_move = x_ego_2d[..., -1, :] - x_ego_2d[..., 0, :]
        ego_move_dis = torch.norm(ego_move, p=2, dim=-1)
        nei_move = x_nei_2d[..., -1, :] - x_nei_2d[..., 0, :]
        nei_move_dis = torch.norm(nei_move, p=2, dim=-1)
        vel_ratio = nei_move_dis / ego_move_dis[:, None]

        group_mask = torch.ones(x_nei_2d.shape[:-2]).to(ego_move.device)
        # group_mask = _group_mask * ((1 - tolerance) < vel_ratio) * (vel_ratio < (1 + tolerance))

        for t in range(x_ego_2d.shape[-2]):
            _vec = x_nei_2d[..., t, :] - x_ego_2d[:, None, t, :]
            _dis = torch.norm(_vec, p=2, dim=-1)
            group_mask = group_mask * (_dis < (1.0 + tolerance[..., :-1]) * ego_move_dis[..., None])
        
        group_mask = group_mask * ((1 - tolerance[..., -1:]) < vel_ratio) * (vel_ratio < (1 + tolerance[..., -1:]))

        trajs_group = (
            x_nei_2d * group_mask[..., None, None]).to(dtype=torch.float32)
        group_num = torch.sum(group_mask, dim=-1)

        return group_mask, trajs_group, group_num
