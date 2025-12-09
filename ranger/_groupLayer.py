"""
@Author: Ziqian Zou
@Date: 2025-11-25 10:28:07
@LastEditors: Ziqian Zou
@LastEditTime: 2025-12-09 15:48:06
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2025 Ziqian Zou, All Rights Reserved.
"""
import numpy as np
import torch

from .__args import RangerArgs

INF = 100000000
MU = 0.00001


class GroupLayer(torch.nn.Module):

    def __init__(self,
                 max_partitions: int,
                 view_angle: float = np.pi,
                 use_socialcircle: int = 1,
                 *args, **kwargs):
        super().__init__()
        self.view_angle = view_angle
        self.use_socialcircle = use_socialcircle
        self.partitions = max_partitions

    def forward(self, trajs: torch.Tensor, nei_trajs: torch.Tensor):
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

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
            torch.sum(torch.abs(nei_trajs), dim=[-1, -2]) < (0.05 * INF)).to(dtype=torch.int32)

        # mask view angle
        view_mask = (torch.abs(nei_dir - obs_dir) <
                     (self.view_angle / 2)).to(dtype=torch.int32)
        left_view_mask = ((nei_dir - obs_dir) > 0).to(dtype=torch.int32)
        right_view_mask = view_mask - left_view_mask

        # mask back angle(places out of the view)
        back_mask = 1 - view_mask

        # all real neighbors in left view, right view and back view
        nei_left = left_view_mask * nei_mask
        nei_right = right_view_mask * nei_mask
        nei_view = nei_left + nei_right
        nei_back = back_mask * nei_mask

        # mask view angle
        view_mask = (torch.abs(nei_dir - obs_dir) <
                     (self.view_angle / 2)).to(dtype=torch.int32)
        left_view_mask = ((nei_dir - obs_dir) > 0).to(dtype=torch.int32)
        right_view_mask = view_mask - left_view_mask

        # calculate neighbors' distance
        dis = torch.norm(nei_posion_vector, dim=-1)

        # calculate neighbors' moving direction
        nei_move_dir_vec = nei_trajs[..., -1:, :] - nei_trajs[..., -2:-1, :]
        nei_move_dir = torch.atan2(
            nei_move_dir_vec[..., 0], nei_move_dir_vec[..., 1])
        nei_move_dir = nei_move_dir % (2*np.pi)
        delta_dir = torch.squeeze(
            (nei_move_dir - obs_dir[:, None, ...]), dim=-1)

        # calculate neighbor's velocity
        velocity = torch.norm(nei_vector, dim=-1)

        if self.use_socialcircle:
            # Angles (the independent variable \theta)
            angle_indices = nei_dir / (2*np.pi/self.partitions)
            angle_indices = angle_indices.to(torch.int32)

            angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

            # Compute the SocialCircle
            social_circle = []
            for ang in range(self.partitions):
                _mask = (angle_indices == ang).to(torch.float32)
                _mask_count = torch.sum(_mask, dim=-1)

                n = _mask_count + 0.0001
                social_circle.append([])

                _velocity = torch.sum(velocity * _mask, dim=-1) / n
                social_circle[-1].append(_velocity)

                _distance = torch.sum(dis * _mask, dim=-1) / n
                social_circle[-1].append(_distance)

                _direction = torch.sum(nei_dir * _mask, dim=-1) / n
                social_circle[-1].append(_direction)

            # Shape of the final SocialCircle: (batch, p, 3)
            social_circle = [torch.stack(i) for i in social_circle]
            social_circle = torch.stack(social_circle)
            social_circle = torch.permute(social_circle, [2, 0, 1])

            if (((m := self.partitions) is not None) and
                    (m > (n := self.partitions))):
                paddings = [0, 0, 0, m - n, 0, 0]
                social_circle = torch.nn.functional.pad(
                    social_circle, paddings)

            return social_circle

        else:

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
            con_back = torch.concat([dis_back[:, None, None]], dim=-1)

            # add right and left
            con = torch.concat([con_right, con_left, con_back], dim=-1)

            return con


class LongTermKernel(torch.nn.Module):

    def __init__(self, group_distance: int,
                 obs_steps: int,
                 *args, **kwargs):
        super().__init__()

        self.group_distance = group_distance
        self.obs_steps = obs_steps

    def forward(self, x_ego_2d: torch.Tensor, x_nei_2d: torch.Tensor):

        # Long term distance between neighbors and obs(ade)
        long_term_dis = x_nei_2d - x_ego_2d[:, None, ...]

        # final step distance(fde)
        final_vec = x_nei_2d[..., -1:, :] - x_ego_2d[:, None, -1:, :]
        
        long_term_sq = torch.sum(long_term_dis ** 2, dim=(-1, -2))
        final_vec_sq = torch.sum(final_vec ** 2, dim=(-1, -2))

        group_mask = (long_term_sq < self.group_distance) & \
                    (final_vec_sq < (self.group_distance / self.obs_steps))
        group_mask = group_mask.int()

        trajs_group = (
            x_nei_2d * group_mask[..., None, None]).to(dtype=torch.float32)
        group_num = torch.sum(group_mask, dim=-1)

        return group_mask, trajs_group, group_num
