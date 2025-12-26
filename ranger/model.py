import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import RangerArgs
from ._groupLayer import INF, GroupLayer, LongTermKernel
from ._trajEncoding import TrajEncoding
from .egoLoss import EgoLoss
from .egoPredictor import EgoPredictor


class RangerModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.ranger_args = self.args.register_subargs(
            RangerArgs, 'ranger_args')
        self.N = 4
        self.ido = self.index_obs
        self.recurrent = False

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Grouplayer
        self.gp = GroupLayer(output_units=self.ranger_args.output_units,
                             view_angle=self.ranger_args.view_angle)

        # Long term kernel function
        self.ltkf = LongTermKernel(
            group_distance=self.ranger_args.group_distance, obs_steps=self.args.obs_frames)

        # Trajectory and feature encoding
        self.te = TrajEncoding(output_units=self.ranger_args.output_units,
                               input_units=self.dim)
        self.te2 = TrajEncoding(output_units=self.ranger_args.output_units * 2,
                                input_units=self.dim)
        self.tse = TrajEncoding(output_units=self.ranger_args.output_units *
                                2, input_units=self.ranger_args.output_units * 3)

        self.concat_fc = layers.Dense(
            self.ranger_args.output_units * 4, self.ranger_args.output_units * 4, activation=torch.nn.Tanh)

        # Linear prediction of obs as the target of transformer
        self.lp = layers.LinearLayerND(
            self.args.obs_frames, self.args.pred_frames, return_full_trajectory=False)

        # Backbone
        self.bb = transformer.Transformer(
            num_layers=4,
            d_model=self.args.feature_dim,
            num_heads=8,
            dff=512,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.pred_frames + self.args.obs_frames,
            include_top=False)

        # Noise encoding
        self.ie = TrajEncoding(self.d, self.d_id)

        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(
            self.ranger_args.output_units * 8, self.ranger_args.generation_num, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(
            self.ranger_args.output_units * 4, self.ranger_args.output_units * 8)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(
            self.ranger_args.output_units * 8, self.ranger_args.output_units * 8, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.ranger_args.output_units * 8,
                                        self.args.pred_frames * self.dim)

        # Ego predictor
        self.ego_predictor = EgoPredictor(
            obs_steps=self.stage_len,
            pred_steps=self.stage_len,
            insights=self.ranger_args.insights_num,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim,
            backbone=self.ranger_args.ego_predictor_type,
            capacity=self.ranger_args.ego_capacity,
            recurrent=self.recurrent)

    def index_obs(self, n: int):
        """
        Decide range of stage.  

        :param n: n starts from 1.
        """
        return n * (self.stage_len)

    @property
    def stage_len(self):
        return self.args.obs_frames // self.N

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        obs_original = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei_original = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # mask neighbors
        obs = obs_original
        nei = nei_original
        nei_mask = (
            torch.sum(torch.abs(nei), dim=[-1, -2]) < (0.05 * INF)).to(dtype=torch.int32)

        nei = torch.concat([obs[..., None, :, :], nei], dim=-3)

        if (t := self.ranger_args.group_type) in [0, 1]:
            # --------------------
            # MARK: - Bare Group Model
            # --------------------
            if t == 0:
                x_nei_pred_3_4 = None
                x_nei_pred_5_6 = None

            # --------------------
            # MARK: - Ego predictor
            # --------------------
            # | 1 | 2 | 3 | 4 |
            # |<---- obs ---->|
            elif t == 1:
                [a, b, c, d] = [1, 2, 2, 3] if self.recurrent \
                    else [0, 2, 0, 2]
                if training:
                    x_nei_pred_3_4 = self.ego_predictor.implement(
                        ego_s1=obs[..., self.ido(a):self.ido(b), :],
                        ego_s2=obs[..., self.ido(c):self.ido(d), :],
                        nei_s1=nei[..., self.ido(a):self.ido(b), :],
                        nei_s2=nei[..., self.ido(c):self.ido(d), :],
                        training=training,
                    )
                    nei_pred_train = x_nei_pred_3_4[..., 1:, :, :, :]

                else:
                    nei_pred_train = None

                x_nei_pred_5_6, x_nei_pred_5_6_not_mean = self.ego_predictor.implement(
                    ego_s1=obs[..., self.ido(a + 2):self.ido(b + 2), :],
                    nei_s1=nei[..., self.ido(a + 2):self.ido(b + 2), :],
                    return_mean=True,
                )

                ego_pred_new = x_nei_pred_5_6[..., 0, :, :]
                nei_pred_new = x_nei_pred_5_6[..., 1:, :, :]

                # Mess up time axis
                nei = torch.concat([
                    nei_original[..., self.ido(2):self.ido(4), :],
                    nei_pred_new], dim=-2
                )
                ego = torch.concat([
                    obs_original[..., self.ido(2):self.ido(4), :],
                    ego_pred_new
                ], dim=-2)

                obs = ego

            elif t == 2:
                raise ValueError

        group_mask, trajs_group, group_num = self.ltkf(obs, nei)

        # Obs trajectory encoding
        f_obs = self.te(obs)

        f_group = self.te(trajs_group)
        f_group = (torch.sum(f_group * group_mask[..., None, None], dim=1) + 1e-8) / \
            (group_num[..., None, None] + 1e-8)

        # Concat obs and nei feature
        f = torch.concat([f_obs, f_group], dim=-1)

        # Compute Conception and padding
        nei_trajs = nei * \
            (1 - group_mask[..., None, None]) + \
            group_mask[..., None, None] * INF

        # -------------
        # Conception Module
        # -------------

        conception_circle = self.gp(obs, nei_trajs)
        f_social = self.tse(conception_circle)
        f_social = torch.repeat_interleave(f_social[..., None, :], torch.tensor(
            f_obs.shape[-2]).to(f_obs.device).to(torch.int32), dim=-2)
        _f = torch.concat([f_social, f], dim=-1)

        f = self.concat_fc(_f)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        obs_lin = self.lp(obs)
        obs_lin = torch.concat([obs, obs_lin], dim=-2)

        # (batch, obs+pred, out_uni * 4)
        f_tran, _ = self.bb(inputs=f, targets=obs_lin, training=training)
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

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y, list(y.shape[:-1]) +
                              [self.args.pred_frames, self.dim])

            all_predictions.append(y)

        Y = torch.concat(all_predictions, dim=-3)

        returns = [
            Y,
        ]

        # Output predictions and labels to compute EgoLoss
        if training:
            returns += [
                nei_original[..., self.index_obs(2):self.index_obs(4), :],
                nei_pred_train,
            ]

        # Visualize ego predictor's outputs
        # This only works in the playground mode
        elif v := self.ranger_args.vis_ego_predictor:
            match v:
                case 1:
                    e = torch.flatten(nei_pred_new, -4, -3)
                case 2:
                    e = nei_pred_new
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

        self.ranger_args = self.args.register_subargs(
            RangerArgs, 'ranger_args')

        if (r := self.ranger_args.ego_capacity) > (m := self.args.max_agents):
            self.log(f'Wrong capacity settings: {r} > {m}!',
                     level='error', raiseError=ValueError)

        if ((self.ranger_args.group_type == 1)
                and (self.ranger_args.ego_predictor_type != 'linear')):
            self.loss.set({l2: self.ranger_args.l2_loss_ratio,
                           EgoLoss: self.ranger_args.ego_loss_ratio})
        else:
            self.loss.set({l2: 1.0})
