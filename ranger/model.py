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
        self.N = 4
        self.ido = self.index_obs
        self.recurrent = False
        self.g = self.ranger_args.output_units

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Grouplayer
        self.gp = GroupLayer(output_units=self.ranger_args.output_units,
                             view_angle=self.ranger_args.view_angle)

        # Grouplayer
        self.gp_2 = GroupLayer(output_units=self.ranger_args.output_units,
                               view_angle=self.ranger_args.view_angle)

        # Long term kernel function
        self.ltkf = LongTermKernel(
            group_distance=self.ranger_args.group_distance, obs_steps=self.args.obs_frames)

        # Trajectory and feature encoding
        self.te = TrajEncoding(output_units=self.ranger_args.output_units,
                               input_units=self.dim)
        self.te2 = TrajEncoding(output_units=self.ranger_args.output_units,
                                input_units=self.dim)
        self.tse = TrajEncoding(output_units=self.ranger_args.output_units *
                                2, input_units=self.ranger_args.output_units * 3)
        self.te3 = torch.nn.Sequential(
            layers.Dense(self.dim, self.g, torch.nn.ReLU),
            layers.Dense(self.g, self.g * 2, torch.nn.ReLU),
            layers.Dense(self.g * 2, self.g, torch.nn.Tanh),
        )

        self.concat_fc = layers.Dense(
            self.ranger_args.output_units * 5, self.ranger_args.output_units * 4, activation=torch.nn.Tanh)

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
            pe_input=self.args.obs_frames * 3,
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
        if self.ranger_args.ego_predictor_type == 'linear':
            e = LinearPrediction
        else:
            e = EgoPredictor

        # Ego predictor
        self.ego_predictor = e(
            obs_steps=self.stage_len,
            pred_steps=self.stage_len,
            insights=self.ranger_args.insights_num,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim,
            backbone=self.ranger_args.ego_predictor_type,
            capacity=self.ranger_args.ego_capacity,
            recurrent=self.recurrent)

        self.tolerance_pred = torch.nn.Sequential(
            layers.Dense(self.g, self.g * 2, activation=torch.nn.ReLU),
            layers.Dense(self.g * 2, self.g * 2, activation=torch.nn.ReLU),
            torch.nn.Flatten(-2, -1),
            layers.Dense(self.g * 2 * self.args.obs_frames,
                         2, activation=torch.nn.Tanh),
            Gate()
        )

        self.fusion = torch.nn.Sequential(
            layers.Dense(self.g * 2, self.g * 2, activation=torch.nn.ReLU),
            layers.Dense(self.g * 2, self.g, activation=torch.nn.Sigmoid)
        )

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
                ego_future = obs
                nei_future = nei

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

                if False:
                    f_obs_0 = self.te(obs)
                    tolerance_0 = self.tolerance_pred(f_obs_0)
                    group_mask_0, trajs_group_0, group_num_0 = self.ltkf(
                        obs, nei, tolerance=tolerance_0)

                # Mess up time axis
                nei_future = torch.concat([
                    nei_original[..., self.ido(2):self.ido(4), :],
                    nei_pred_new], dim=-2
                )
                ego_future = torch.concat([
                    obs_original[..., self.ido(2):self.ido(4), :],
                    ego_pred_new
                ], dim=-2)

                # obs = ego

            elif t == 2:
                raise ValueError

        f_obs = self.te(ego_future)
        tolerance = self.tolerance_pred(f_obs)

        # print tolerance
        self.get_top_manager().log(str(tolerance))

        # divide in-group and out-of-group through mess up trajs
        group_mask, trajs_group_mess, group_num = self.ltkf(
            ego_future, nei_future, tolerance=tolerance)

        if False:
            group_indices = torch.nonzero(group_mask, as_tuple=True)
            group_indices_0 = torch.nonzero(group_mask_0, as_tuple=True)
            group_trajs = trajs_group_mess[group_indices][None]
            group_trajs_0 = trajs_group_0[group_indices_0][None]
            _group_trajs = torch.flatten(group_trajs, -3, -2)
            _group_trajs_0 = torch.flatten(group_trajs_0, -3, -2)
            h = _group_trajs.shape[1]
            i = _group_trajs_0.shape[1]
            if h * i == 0:
                return torch.zeros([1, 1, 2])

            m = torch.zeros_like(
                _group_trajs[:, -1:, :].expand(-1, abs((h) - (i)), -1))

            if h > i:
                _group_trajs_0 = torch.concat([_group_trajs_0, m], dim=-2)
            elif h < i:
                _group_trajs = torch.concat([_group_trajs, m], dim=-2)

            return torch.concat([_group_trajs, _group_trajs_0], dim=-3)[None]
        # Obs trajectory encoding

        f_group = self.te2(trajs_group_mess) * group_mask[..., None, None]
        f_group = torch.max(f_group, dim=-3)[0] * (1.0 / (1.0 + tolerance[..., None, :-1]))
        f_ego = torch.concat([f_obs * (1.0 + tolerance[..., None, -1:]), f_group], dim=-1)

        # Compute Conception and padding
        nei_trajs = nei_original * \
            (1 - group_mask[..., None, None]) + \
            group_mask[..., None, None] * INF

        # -------------
        # Conception Module
        # -------------
        out_group_mask = nei_mask * (1 - group_mask)
        out_nei_trajs = nei_trajs * out_group_mask[..., None, None]

        f_out_group = self.gp(obs, out_nei_trajs, reshape=True)
        f_s = repeat(f_out_group[..., None, :], f_ego.shape[-2], dim=-2)

        # def stat(name, t):
        #     with torch.no_grad():
        #         print(name, t.abs().mean().item(), t.std().item(), t.abs().max().item())
        # stat("f_obs", f_obs)
        # stat("f_group", f_group)
        # stat("f_s", f_s)
        f_s = f_s * (1 / (1 + tolerance[..., None, -1:])) * (1 / (1 + tolerance[..., None, :-1]))
        f = torch.concat([f_ego, f_s], dim=-1)

        f = self.concat_fc(f)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        pred_linear = self.lp(obs)
        obs_lin = torch.concat([obs, pred_linear], dim=-2)

        # (batch, obs+pred, out_uni * 4)
        f_tran, _ = self.bb(inputs=f.reshape(
            f.shape[0], -1, f.shape[-1]), targets=obs_lin, training=training)
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
            Y + pred_linear[..., None, :, :],
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
