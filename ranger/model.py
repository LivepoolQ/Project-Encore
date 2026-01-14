import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import RangerArgs
from ._groupLayer import INF, MU, GroupLayer, LongTermKernel
from ._trajEncoding import TrajEncoding
from .egoLoss import EgoLoss
from .egoPredictor import EgoPredictor, LinearPrediction
from .utils import repeat, Gate


class RangerModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.ranger_args = self.args.register_subargs(
            RangerArgs, 'ranger_args')
        self.r = self.ranger_args
        self.g = self.d
        self.t_h = self.r.ego_t_h
        self.t_f = self.r.ego_t_f

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Grouplayer: perceive out-of-group agents
        self.gp = GroupLayer(output_units=self.d,
                             view_angle=self.r.view_angle)

        # Long term kernel function
        self.ltkf = LongTermKernel(
            group_distance=self.r.group_distance,
            obs_steps=self.args.obs_frames
        )

        # Trajectory and feature encoding
        # Encode ego's obs
        self.te = TrajEncoding(output_units=self.d,
                               input_units=self.dim)

        # Encode all obs in the same group
        self.te2 = TrajEncoding(output_units=self.d,
                                input_units=self.dim)

        # Concat all ego, group, out-of-group agents feature and encode
        self.concat_fc = layers.Dense(
            self.d * 5,
            self.d * 4,
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
            d_model=self.d * 4,
            num_heads=8,
            dff=512,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.pred_frames + self.args.obs_frames,
            include_top=False
        )

        # Noise encoding
        self.ie = TrajEncoding(self.d * 4, self.d_id)

        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(self.d * 8,
                                  self.r.generation_num,
                                  torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d * 4, self.d * 8)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d * 8, self.d * 8, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.d * 8,
                                        self.args.pred_frames * self.dim)

        # `linear` type is only used in ablation
        if self.r.ego_predictor_type == 'linear' or self.r == 0:
            e = LinearPrediction
        else:
            e = EgoPredictor

        # Ego predictor
        self.ego_predictor = e(
            obs_steps=self.t_h,
            pred_steps=self.t_f,
            insights=self.r.insights_num,
            traj_dim=self.dim,
            feature_dim=self.d * 4,
            backbone=self.r.ego_predictor_type,
            capacity=self.r.ego_capacity,
        )

        # Tolerance prediction network
        self.tolerance_pred = torch.nn.Sequential(
            layers.Dense(self.g, self.g * 2, activation=torch.nn.ReLU),
            layers.Dense(self.g * 2, self.g * 2, activation=torch.nn.ReLU),
            torch.nn.Flatten(-2, -1),
            layers.Dense(self.g * 2 * self.args.obs_frames,
                         2, activation=torch.nn.Tanh),
            Gate()
        )

    @property
    def d(self) -> int:
        """
        Basic output units
        """
        return self.ranger_args.output_units

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        obs_original = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei_original = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # mask neighbors
        obs = obs_original
        nei = nei_original
        nei_mask = torch.sum(nei.abs(), dim=[-1, -2]) < (0.05 * INF)
        nei_mask = nei_mask.to(dtype=torch.int32)

        # pack ego and nei for faster inference
        nei = torch.concat([obs[..., None, :, :], nei], dim=-3)

        # `t` accepts three values:
        # - `0`: Vanilla;
        # - `1`: TODO Model;
        # - `2`: TODO.
        if (t := self.r.group_type) in [0, 1]:
            # ------------------------
            # MARK: - Bare Group Model
            # ------------------------
            if t == 0:
                y_ego_train = None
                y_ego = None
                egos = obs
                neis = nei_original

            # --------------------
            # MARK: - Ego predictor
            # --------------------

            # |<---- obs ---->|<-------- pred -------->|
            #         |<-t_h->|<-t_f->| <- inference
            # |<-t_h->|<-t_f->|         <- train
            elif t == 1:
                if training:
                    y_ego_train = self.ego_predictor.implement(
                        ego_s1=obs[..., -(self.t_h + self.t_f):-self.t_f, :],
                        nei_s1=nei[..., -(self.t_h + self.t_f):-self.t_f, :],
                        training=training
                    )
                    nei_pred_train = y_ego_train[..., 1:, :, :, :]

                else:
                    nei_pred_train = None

                y_ego_packed, _ = self.ego_predictor.implement(
                    ego_s1=obs[..., -self.t_h:, :],
                    nei_s1=nei[..., -self.t_h:, :],
                    return_mean=True,
                )

                y_ego = y_ego_packed[..., 0, :, :]
                y_nei = y_ego_packed[..., 1:, :, :]

                # Mess up time axis
                neis = torch.concat([
                    nei_original[..., -self.t_h:, :],
                    y_nei], dim=-2
                )
                egos = torch.concat([
                    obs_original[..., -self.t_h:, :],
                    y_ego
                ], dim=-2)

            elif t == 2:
                raise NotImplementedError

        # Encode ego's imagined obs and tolerance
        f_obs = self.te(egos)
        tolerance = self.tolerance_pred(f_obs)

        # grouping imagined trajectories
        group_mask, trajs_group_mess, _ = self.ltkf(
            egos, neis, tolerance=tolerance)

        # imagined trajectory encoding (in-group)
        f_group = self.te2(trajs_group_mess) * group_mask[..., None, None]
        f_group = torch.max(
            f_group, dim=-3)[0] * (1.0 / (1.0 + tolerance[..., None, :1]))
        f_ego = torch.concat(
            [f_obs * (1.0 + tolerance[..., None, -1:]), f_group], dim=-1)

        # apply mask
        nei_trajs = ((1 - group_mask[..., None, None]) * nei_original +
                     group_mask[..., None, None] * INF)

        # -------------
        # Conception Module
        # -------------
        out_group_mask = nei_mask * (1 - group_mask)
        out_nei_trajs = nei_trajs * out_group_mask[..., None, None]

        # imagined trajectory encoding (out-of-group)
        f_out_group = self.gp(obs, out_nei_trajs, reshape=True)
        f_s = repeat(f_out_group[..., None, :], f_ego.shape[-2], dim=-2)

        # use tolerance characterize out-of-group feature
        f_s = (f_s *
               (1 / (1 + tolerance[..., None, -1:])) *
               (1 / (1 + tolerance[..., None, :1])))

        f = torch.concat([f_ego, f_s], dim=-1)
        f = self.concat_fc(f)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        pred_linear = self.lp(obs)
        pred_linear = pred_linear - \
            pred_linear[..., None, self.args.obs_frames - 1, :]
        # obs_lin = torch.concat([obs, pred_linear], dim=-2)

        # (batch, obs + pred, out_uni * 4)
        f_tran, _ = self.bb(
            inputs=f,
            targets=pred_linear,
            training=training
        )

        # slice the prediction time steps
        # (batch, pred, out_uni * 4)
        f_tran = f_tran[:, self.args.obs_frames:, ...]

        # Prediction
        for _ in range(repeats):
            # Assign random ids and embedding
            z = torch.normal(mean=0, std=1, size=list(
                f_tran.shape[:-1]) + [self.d_id])
            # (batch, pred, out_uni * 4)
            f_z = self.ie(z.to(obs.device))

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
                nei_original[..., -self.t_f:, :],
                nei_pred_train,
            ]

        # Visualize ego predictor's outputs
        # This only works in the playground mode
        elif v := self.r.vis_ego_predictor:
            match v:
                case 1:
                    e = torch.flatten(y_nei, -4, -3)
                case 2:
                    e = y_nei
                case _:
                    self.log(f'Wrong `vis_ego_predictor` value recevied: {v}!',
                             level='error', raiseError=ValueError)

            returns[0] = e

        return returns


class Ranger(Structure):
    MODEL_TYPE = RangerModel

    def __init__(self, args=None,
                 manager=None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        self.r = self.args.register_subargs(
            RangerArgs, 'ranger_args')

        if (r := self.r.ego_capacity) > (m := self.args.max_agents):
            self.log(f'Wrong capacity settings: {r} > {m}!',
                     level='error', raiseError=ValueError)

        if ((self.r.group_type == 1)
                and (self.r.ego_predictor_type != 'linear')):
            self.loss.set({l2: self.r.l2_loss_ratio,
                           EgoLoss: self.r.ego_loss_ratio})
        else:
            self.loss.set({l2: 1.0})
