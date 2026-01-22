"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Ziqian Zou
@LastEditTime: 2026-01-09 16:20:51
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

from typing import overload

import torch

from qpid.model import layers, transformer
from qpid.utils import get_mask

from .linearDiffEncoding import LinearDiffEncoding
from .reverberationTransform import ReverberationTransform
from .utils import KernelLayer


class EgoPredictor(torch.nn.Module):
    """
    EgoPredictor
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 traj_dim: int,
                 feature_dim: int,
                 backbone: str,
                 capacity: int = -1,
                 *args, **kwargs):

        super().__init__()

        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d_traj = traj_dim
        self.d = feature_dim

        self.insights = insights
        self.backbone = backbone

        self.capacity = capacity

        # Simple trajectory predictor, similar to the reverberation transform
        self.outer = layers.OuterLayer(self.t_h, self.t_h)
        self.reverberation_predictor = KernelLayer(self.d, self.d, self.t_f)
        self.insight_predictor = KernelLayer(self.d, self.d, self.insights)

        self.traj_embed = layers.Dense(
            input_units=self.d_traj,
            output_units=self.d,
            activation=torch.nn.Tanh)

        if self.backbone == 'tran':
            # Simple trajectory encoder and decoder
            self.encoder = transformer.Transformer(
                num_layers=2,
                num_heads=2,
                d_model=self.d,
                dff=self.d,
                pe_input=self.t_h,
                pe_target=self.t_h,
                input_vocab_size=self.d_traj,
                target_vocab_size=self.d_traj,
                include_top=False)

        elif self.backbone == 'linear':
            self.encoder = layers.LinearLayerND(
                obs_frames=self.t_h,
                pred_frames=self.t_f,
                return_full_trajectory=False)

        elif self.backbone == 'fc':
            self.encoder = torch.nn.Sequential(
                layers.Dense(self.d, self.d, torch.nn.ReLU),
                layers.Dense(self.d, self.d, torch.nn.ReLU),
                layers.Dense(self.d, self.d, torch.nn.ReLU),
                layers.Dense(self.d, self.d, torch.nn.ReLU),
                layers.Dense(self.d, self.d, torch.nn.ReLU),
                layers.Dense(self.d, self.d, torch.nn.Tanh))

        else:
            raise ValueError('Wrong ego predictor backbone type!')

        self.decoder = layers.Dense(self.d, self.d_traj)

        self.noise_embedding = layers.Dense(32, self.d, torch.nn.Tanh)

        self.concat_fc = layers.Dense(self.d * 2, self.d, torch.nn.Tanh)

        # Linear prediction layer (for out-of-capacity neighbors)
        self.linear_pred = layers.LinearLayerND(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
        )

        self.rev = ReverberationTransform(
            historical_steps=self.t_h,
            future_steps=self.t_f
        )

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers('none')
        self.tlayer = t_type((self.t_h, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Linear difference encoding (embedding)
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
            output_units=self.d,
            transform_layer=self.tlayer,
        )

    def forward(self, ego_traj: torch.Tensor, nei_trajs: torch.Tensor, training=None):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        # Repeat and concat ego and neighbors' obs
        x_ego = ego_traj
        x_nei = nei_trajs
        max_nei_count = x_nei.shape[-3]
        _x_ego = x_ego[..., None, :, :].expand(
            *x_ego.shape[:-2],
            max_nei_count,
            *x_ego.shape[-2:],
        )

        _x = torch.concat([_x_ego, x_nei], dim=-2)

        # Speed up inference #1: Remove all-empty neighbors
        # Compute max neighbor count within the batch
        valid_mask = get_mask(torch.sum(torch.abs(x_nei), dim=[-1, -2]))

        # Speed up inference #2: Limit neighbor numbers
        if self.capacity > 0:
            # Compute relative distance (at the last obs step)
            d = torch.norm(x_ego[..., -1:, :] - x_nei[..., -1, :],
                           p=2, dim=-1)
            idx = torch.topk(d, self.capacity, dim=-1, largest=False).indices

            # Compute the min-distance neighbor mask
            cap_mask = torch.zeros_like(d)
            cap_mask = torch.scatter(cap_mask, -1, idx, 1)
        else:
            cap_mask = 1

        # Compute final mask
        final_mask = valid_mask * cap_mask

        # Get neighbors to be considered
        indices = torch.nonzero(final_mask, as_tuple=True)
        x_picked = _x[indices]

        _x_ego = x_picked[..., :self.t_h, :]
        _x_nei = x_picked[..., self.t_h:, :]

        # Concat ego and neighbors' trajectories into a `big batch`
        b = _x_ego.shape[0]
        x_packed = torch.concat([_x_ego, _x_nei], dim=0)

        # Move the last obs point to (0, 0)
        ref = x_packed[..., -1:, :]         # (b*2, t_h, dim)
        x_packed = x_packed - ref

        if ((ego_traj.shape[-2] != self.t_h) or
                (nei_trajs.shape[-2] != self.t_h)):
            raise ValueError('Wrong trajectory lengths!')

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Encode features together
        # Including the insight feature and neighbor features
        _f, x_linear, y_linear = self.linear_diff(x_packed)
        x_diff = x_packed - x_linear

        if isinstance(self.encoder, transformer.Transformer):
            f_pack, _ = self.encoder.forward(
                inputs=_f, targets=x_diff, training=training)
        else:
            f_pack = self.encoder(_f)

        # Add noise
        z = torch.normal(mean=0, std=1,
                         size=list(f_pack.shape[:-1]) + [32])
        f_z = self.noise_embedding(z.to(f_pack.device))

        # -> (b*2, T_h, d)
        f_final = torch.concat([f_pack, f_z], dim=-1)
        f_final = self.concat_fc(f_final)

        # Unpack features
        f_insight = f_final[:b, :, :]
        f_nei = f_final[b:, :, :]

        # Compute kernels
        # (batch, nei, t_h, t_f)
        rev_kernel = self.reverberation_predictor(f_nei)

        # (batch, 1, t_h, insights)
        ins_kernel = self.insight_predictor(f_insight)

        # Predict (like reverberation transform)
        f_rev = self.rev(f_nei, rev_kernel, ins_kernel)

        # Decode predictions
        # (batch, nei, insights, t_f, dim)
        pred = self.decoder(f_rev)

        # Move back predictions
        pred = pred + ref[b:, None, :, :] + y_linear[b:, None, :, :]

        # Run linear prediction for un-masked neighbors
        y_nei_base = self.linear_pred(x_nei)
        y_nei_base = y_nei_base[..., None, :, :].expand(
            *y_nei_base.shape[:-2],
            self.insights,
            *y_nei_base.shape[-2:],
        )

        y = y_nei_base.clone()
        y[indices] = pred

        return y

    @overload
    def implement(self, ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None) -> torch.Tensor:
        """
        Foward ego predictor, return the prediction *as is*.
        """
        ...

    @overload
    def implement(self, ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None,
                  return_mean: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Foward ego predictor, return the *mean* prediction for each neighbor.
        """
        ...

    def implement(self, ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None,
                  return_mean: bool = False):

        if self.backbone == 'linear':
            y = self.encoder(nei_s1)

            return y if return_mean else y[..., None, :, :]

        # Recurrent prediction: 2 -> 3
        x_nei_pred_s2 = self(
            ego_traj=ego_s1,
            nei_trajs=nei_s1,
            training=training,
        )

        if return_mean:
            return torch.mean(x_nei_pred_s2, dim=-3), x_nei_pred_s2
        else:
            return x_nei_pred_s2


class LinearPrediction(torch.nn.Module):

    def __init__(self,
                 obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 *args, **kwargs):

        super().__init__()

        self.insights = insights
        self.t_h = obs_steps
        self.t_f = pred_steps

        self.encoder = layers.LinearLayerND(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
            return_full_trajectory=False)

    def forward(self, nei_trajs: torch.Tensor):

        y_nei = self.encoder(nei_trajs)
        y_nei_not_mean = y_nei[..., None, :, :].expand(
            *y_nei.shape[:-2],
            self.insights,
            *y_nei.shape[-2:],
        )

        return y_nei, y_nei_not_mean

    @overload
    def implement(self, ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None) -> torch.Tensor:
        """
        Foward ego predictor, return the prediction *as is*.
        """
        ...

    @overload
    def implement(self, ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None,
                  return_mean: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Foward ego predictor, return the *mean* prediction for each neighbor.
        """
        ...

    def implement(self,
                  ego_s1: torch.Tensor,
                  nei_s1: torch.Tensor,
                  training=None,
                  return_mean: bool = False,
                  *args, **kwargs):

        y, y_not_mean = self(nei_s1)

        if return_mean:
            return y, y_not_mean
        else:
            return y_not_mean
