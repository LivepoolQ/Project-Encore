import numpy as np
import torch
from .egoPredictor import EgoPredictor, LinearPrediction
from .utils import Gate

from qpid.model import layers


class GroupingKernel(torch.nn.Module):
    def __init__(self,
                 traj_dim: int,
                 feature_dim: int,
                 obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 backbone: str,
                 use_mixed: int,
                 fix_dis_anchor: int,
                 fix_speed_anchor: int,
                 set_anchor: int,
                 set_dis_anchor=None,
                 set_speed_anchor=None,
                 ego_capacity: int = -1,
                 ego_t_h: int = -1,
                 ego_t_f: int = -1,
                 previews_only: int = 0,
                 vis_anchors: int = 0,
                 disable_dis_anchor: int = 0,
                 disable_speed_anchor: int = 0,
                 current_only: int = 0,
                 set_grouping_ratio: float = -1,
                 inject_noise_std: float = 0.0,
                 swap_prediction_ratio: float = 0.0,
                 flip_prediction_ratio: float = 0.0,
                 *args, **kwargs):
        super().__init__()

        self.traj_dim = traj_dim
        self.feature_dim = feature_dim
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.insights = insights
        self.backbone = backbone
        self.ego_capacity = ego_capacity
        self.t_h = ego_t_h
        self.t_f = ego_t_f
        self.use_mixed = use_mixed
        self.fix_dis = fix_dis_anchor
        self.fix_speed = fix_speed_anchor
        self.set_anchor = set_anchor
        self.set_dis_anchor = set_dis_anchor
        self.set_speed_anchor = set_speed_anchor
        self.previews_only = previews_only
        self.vis_anchors = vis_anchors
        self.disable_dis_anchor = disable_dis_anchor
        self.disable_speed_anchor = disable_speed_anchor
        self.current_only = current_only
        self.set_grouping_ratio = set_grouping_ratio
        self.inject_noise_std = inject_noise_std
        self.swap_prediction_ratio = swap_prediction_ratio
        self.flip_prediction_ratio = flip_prediction_ratio

        # # Encode ego's obs
        self.ego_te = torch.nn.Sequential(
            layers.Dense(input_units=self.traj_dim,
                         output_units=self.feature_dim,
                         activation=torch.nn.ReLU),
            layers.Dense(input_units=self.feature_dim,
                         output_units=self.feature_dim,
                         activation=torch.nn.Tanh),
        )

        # Socialality prediction network
        self.socialality_pred = torch.nn.Sequential(
            layers.Dense(self.feature_dim,
                         self.feature_dim * 2,
                         activation=torch.nn.ReLU),
            layers.Dense(self.feature_dim * 2,
                         self.feature_dim * 2,
                         activation=torch.nn.ReLU),
            torch.nn.Flatten(-2, -1),
            layers.Dense(self.feature_dim * 2 * self.obs_steps,
                         2,
                         activation=torch.nn.Tanh),
            Gate()
        )

        # Socialality Kernel (grouping)
        self.grouping = SocialalityKernel(
            obs_steps=self.obs_steps,
            current_only=self.current_only,
        )

        # `linear` type is only used in ablation
        if self.backbone == 'linear':
            e = LinearPrediction
        else:
            e = EgoPredictor

        # Ego predictor (mixed time)
        self.ego_pred = e(
            obs_steps=self.t_h,
            pred_steps=self.t_f,
            insights=self.insights,
            traj_dim=self.traj_dim,
            feature_dim=self.feature_dim,
            backbone=self.backbone,
            capacity=self.ego_capacity,
        )

    def forward(self, ego_traj: torch.Tensor, nei_trajs: torch.Tensor, training=None):

        if self.use_mixed:
            # ---------------------
            # MARK: - Ego predictor
            # ---------------------
            # pack ego and nei for faster inference
            nei_packed = torch.concat(
                [ego_traj[..., None, :, :], nei_trajs], dim=-3)

            # |<---- obs ---->|<-------- pred -------->|
            #         |<-t_h->|<-t_f->| <- inference
            # |<-t_h->|<-t_f->|         <- train
            if training:
                y_ego_train = self.ego_pred.implement(
                    ego_s1=ego_traj[..., -(self.t_h + self.t_f):-self.t_f, :],
                    nei_s1=nei_packed[..., -
                                      (self.t_h + self.t_f):-self.t_f, :],
                    training=training
                )
                nei_pred_train = y_ego_train[..., 1:, :, :, :]

            else:
                nei_pred_train = None

            y_ego_packed, _ = self.ego_pred.implement(
                ego_s1=ego_traj[..., -self.t_h:, :],
                nei_s1=nei_packed[..., -self.t_h:, :],
                return_mean=True,
            )

            if self.swap_prediction_ratio > 0.0:
                mask = torch.rand(y_ego_packed.shape[:-2],
                                  device=y_ego_packed.device) < self.swap_prediction_ratio
                rand_indices = torch.argsort(torch.rand(
                    y_ego_packed.shape[:-2], device=y_ego_packed.device), dim=-1)
                rand_indices_expanded = rand_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    *rand_indices.shape, y_ego_packed.shape[-2], y_ego_packed.shape[-1])
                y_swapped = torch.gather(
                    y_ego_packed, -3, rand_indices_expanded)
                mask = mask[..., None, None]
                y_ego_packed = torch.where(mask, y_swapped, y_ego_packed)

            if self.flip_prediction_ratio > 0.0:
                mask = torch.rand(
                    y_ego_packed.shape[:-2], device=y_ego_packed.device) < self.flip_prediction_ratio
                mask = mask[..., None, None]
                last_obs_packed = nei_packed[..., -1:, :]
                y_flipped = 2 * last_obs_packed - y_ego_packed
                y_ego_packed = torch.where(mask, y_flipped, y_ego_packed)

            y_ego = y_ego_packed[..., 0, :, :]
            y_nei = y_ego_packed[..., 1:, :, :]

            if self.inject_noise_std > 0.0:
                y_ego = y_ego + torch.randn_like(y_ego) * self.inject_noise_std
                y_nei = y_nei + torch.randn_like(y_nei) * self.inject_noise_std

            # Mix up time axis
            nei_trajs = torch.concat([
                nei_trajs[..., -self.t_h:, :],
                y_nei], dim=-2
            )
            ego_traj = torch.concat([
                ego_traj[..., -self.t_h:, :],
                y_ego
            ], dim=-2)

            if self.previews_only:
                y_ego_preview_packed, _ = self.ego_pred.implement(
                    ego_s1=y_ego,
                    nei_s1=y_ego_packed,
                    return_mean=True,
                )
                y_ego_preview = y_ego_preview_packed[..., 0, :, :]
                y_nei_preview = y_ego_preview_packed[..., 1:, :, :]

                # Further extend time axis
                nei_trajs = torch.concat([
                    y_nei,
                    y_nei_preview], dim=-2
                )
                ego_traj = torch.concat([
                    y_ego,
                    y_ego_preview
                ], dim=-2)

        else:
            if self.previews_only:
                raise ValueError(
                    'This args can only be used when --use_mixed_trajectory 1.')

            nei_pred_train = nei_trajs[..., -self.t_f:, :]
            y_nei = None
        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Encode ego
        f_ego = self.ego_te(ego_traj)  # (bs, obs_steps, d)

        # predict socialality
        socialality = self.socialality_pred(f_ego)  # (bs, 2)

        # ------------------------------------
        # MARK: - Socialality Anchor Ablations
        # ------------------------------------
        # Only used in ablations
        need_modify = (self.fix_dis
                       or self.fix_speed
                       or self.set_anchor
                       or self.disable_dis_anchor
                       or self.disable_speed_anchor)
        if need_modify:
            socialality = socialality.clone()

            # -------------------------
            # Set anchor to constant
            # -------------------------
            if self.set_anchor:
                # set distance anchor value
                if self.set_dis_anchor != -1:
                    socialality[..., 0] = self.set_dis_anchor

                # set speed anchor value
                if self.set_speed_anchor != -1:
                    socialality[..., 1] = self.set_speed_anchor

            # -------------------------
            # Fix anchor (stop-grad)
            # -------------------------
            # only detach if this anchor is NOT already set to constant
            if self.fix_dis and not (self.set_anchor and self.set_dis_anchor != -1):
                socialality[..., 0] = torch.detach(socialality[..., 0])

            if self.fix_speed and not (self.set_anchor and self.set_speed_anchor != -1):
                socialality[..., 1] = torch.detach(socialality[..., 1])

            # -------------------------
            # Disable anchor
            # -------------------------
            # can only disable one of the two anchors, both anchors being disabled means that
            # the whole grouping kernel is disabled, this condition is controlled by another
            # ablation arg TODO
            if self.disable_dis_anchor and self.disable_speed_anchor:
                raise ValueError('distance anchor and speed anchor \
                                 cannot be disabled at the same time!')

            if self.disable_dis_anchor and not self.disable_speed_anchor:
                socialality = socialality[..., 1:]

            if not self.disable_dis_anchor and self.disable_speed_anchor:
                socialality = socialality[..., :1]

        # --------------------------------
        # Socialality achors visualization
        # --------------------------------
        if self.vis_anchors:
            # Resort trajectories according to the last point.
            # Here `x_ego` is actually `x_nei` for the ego agent.
            d = torch.norm(nei_trajs[..., self.t_h, :], p=2, dim=-1)
            batch_id = d.argsort()
            batch_num = torch.arange(nei_trajs.shape[0])[:, None]
            nei_trajs = nei_trajs[batch_num, batch_id]

            # Assign labels for better visualization
            M, N = nei_trajs.shape[:2]
            IDs = np.array([[f'b{m}_n{n}' for n in range(N)]
                           for m in range(M)])

            from qpid.utils import get_mask
            # Remove invalid trajectories.
            mask = get_mask(nei_trajs.abs().sum([-1, -2]))
            idx = torch.where(mask.bool())

            IDs = list(IDs[idx])

            from .utils import vis_socialality
            vis_socialality(socialality, IDs)

        # grouping agents using predicted socialality factor
        group_mask, trajs_group, _ = self.grouping(ego_traj,
                                                   nei_trajs,
                                                   socialality,
                                                   self.disable_dis_anchor,
                                                   self.disable_speed_anchor)

        # --------------------------
        # MARK: - Grouping Ablations
        # --------------------------
        # Only used in ablations
        if (r := self.set_grouping_ratio) >= 0:
            group_mask = torch.rand(group_mask.shape) < r
            group_mask = group_mask.to(torch.float32).to(ego_traj.device)

        return group_mask, trajs_group, f_ego, socialality, nei_pred_train, y_nei, nei_trajs


class SocialalityKernel(torch.nn.Module):

    def __init__(self,
                 obs_steps: int,
                 current_only: int,
                 *args, **kwargs):
        super().__init__()

        self.obs_steps = obs_steps
        self.current_only = current_only

    def forward(self,
                x_ego_2d: torch.Tensor,
                x_nei_2d: torch.Tensor,
                tolerance: torch.Tensor,
                disable_dis: int,
                disable_speed: int):

        ego_move = x_ego_2d[..., -1, :] - x_ego_2d[..., 0, :]
        ego_move_dis = torch.norm(ego_move, p=2, dim=-1)
        nei_move = x_nei_2d[..., -1, :] - x_nei_2d[..., 0, :]
        nei_move_dis = torch.norm(nei_move, p=2, dim=-1)
        vel_ratio = nei_move_dis / ego_move_dis[:, None]

        group_mask = torch.ones(x_nei_2d.shape[:-2]).to(ego_move.device)

        if not disable_dis and not disable_speed:
            for t in range(x_ego_2d.shape[-2]):
                _vec = x_nei_2d[..., t, :] - x_ego_2d[:, None, t, :]
                _dis = torch.norm(_vec, p=2, dim=-1)
                group_mask = group_mask * \
                    (_dis < (1.0 + tolerance[..., :-1])
                     * ego_move_dis[..., None])

            group_mask = group_mask * \
                ((1 - tolerance[..., -1:]) < vel_ratio) * \
                (vel_ratio < (1 + tolerance[..., -1:]))

        if disable_dis and not disable_speed:
            group_mask = group_mask * \
                ((1 - tolerance[..., -1:]) < vel_ratio) * \
                (vel_ratio < (1 + tolerance[..., -1:]))

        if disable_speed and not disable_dis:
            for t in range(x_ego_2d.shape[-2]):
                _vec = x_nei_2d[..., t, :] - x_ego_2d[:, None, t, :]
                _dis = torch.norm(_vec, p=2, dim=-1)
                group_mask = group_mask * \
                    (_dis < (1.0 + tolerance)
                     * ego_move_dis[..., None])

        if self.current_only:
            _vec = x_nei_2d[..., -1, :] - x_ego_2d[:, None, -1, :]
            _dis = torch.norm(_vec, p=2, dim=-1)
            group_mask = torch.ones(x_nei_2d.shape[:-2]).to(ego_move.device) * \
                (_dis < (1.0 + tolerance[..., :-1])
                    * ego_move_dis[..., None])

        trajs_group = (
            x_nei_2d * group_mask[..., None, None]).to(dtype=torch.float32)
        group_num = torch.sum(group_mask, dim=-1)

        return group_mask, trajs_group, group_num


class LongTermKernel(torch.nn.Module):
    def __init__(self,
                 threshold: float,
                 *args,
                 **kwargs):

        super().__init__()

        self.threshold = threshold

    def forward(self, x_ego: torch.Tensor, x_nei: torch.Tensor):
        """
        x_ego and x_nei consist of selected only time steps.
        """
        return
