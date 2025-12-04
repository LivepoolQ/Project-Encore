import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure

from .__args import RangerArgs
from ._groupLayer import GroupLayer, LongTermKernel
from ._trajEncoding import TrajEncoding
from .feedbackLayer import ExpectationLayer, PerceiveLayer

INF = 100000000


class RangerModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.ranger_args = self.args.register_subargs(
            RangerArgs, 'ranger_args')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Grouplayer
        self.gp = GroupLayer(view_angle=self.ranger_args.view_angle, 
                             use_socialcircle=self.ranger_args.use_socialcircle,
                             max_partitions=self.ranger_args.partitions)

        # Long term kernel function
        self.ltkf = LongTermKernel(
            group_distance=self.ranger_args.group_distance, obs_steps=self.args.obs_frames)

        # Trajectory and feature encoding
        self.te = TrajEncoding(output_units=self.ranger_args.output_units,
                               input_units=self.dim)
        self.te_g = TrajEncoding(output_units=self.ranger_args.output_units,
                                 input_units=self.dim)
        self.te2 = TrajEncoding(output_units=self.ranger_args.output_units * 2,
                                input_units=self.dim)
        self.tse = TrajEncoding(
            output_units=self.ranger_args.output_units * 2, input_units=7)
        self.tse_2 = TrajEncoding(
            output_units=self.ranger_args.output_units * 2, input_units=3)

        self.concat_fc = layers.Dense(self.ranger_args.output_units * 4, 
                                      self.ranger_args.output_units * 4, 
                                      activation=torch.nn.Tanh)

        # Linear prediction of obs as the target of transformer
        self.lp = layers.LinearLayerND(
            self.args.obs_frames, self.args.pred_frames, return_full_trajectory=False)

        # Backbone
        self.bb = transformer.Transformer(
            num_layers=4,
            d_model=self.args.feature_dim,
            num_heads=self.args.obs_frames,
            dff=self.args.feature_dim * 4,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            pe_input=self.args.obs_frames,
            pe_target=self.args.pred_frames + self.args.obs_frames,
            include_top=False
        )

        if self.ranger_args.enable_per_tran:
            self.bb_lite = transformer.Transformer(
                num_layers=4,
                d_model=self.args.feature_dim//2,
                num_heads=self.args.obs_frames,
                dff=self.args.feature_dim*2,
                input_vocab_size=self.args.feature_dim//2,
                target_vocab_size=self.args.feature_dim//2,
                pe_input=self.args.obs_frames,
                pe_target=self.args.obs_frames,
                include_top=False
            )

        if self.ranger_args.enable_exp_tran:
            self.bb_lite_2 = transformer.Transformer(
                num_layers=4,
                d_model=self.args.feature_dim//2,
                num_heads=self.args.obs_frames,
                dff=self.args.feature_dim*2,
                input_vocab_size=self.args.feature_dim//2,
                target_vocab_size=self.args.feature_dim//2,
                pe_input=self.args.obs_frames,
                pe_target=self.args.obs_frames,
                include_top=False
            )

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

        # Expectation and Perceive layer
        self.exp = ExpectationLayer(self.ranger_args.output_units * 4, 
                                    self.ranger_args.output_units * 4, 
                                    self.ranger_args.output_units * 2, 
                                    self.ranger_args.use_activation)
        
        self.per = PerceiveLayer(self.ranger_args.output_units * 4, 
                                    self.ranger_args.output_units * 4, 
                                    self.ranger_args.output_units * 2, 
                                    self.ranger_args.use_activation)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        group_mask, trajs_group, group_num = self.ltkf(obs, nei)

        if self.ranger_args.use_group:

            # Obs trajectory encoding
            f_obs = self.te(obs)

            f_group = self.te_g(trajs_group)
            f_group = (torch.sum(f_group * group_mask[..., None, None], dim=1) + 1e-8) / \
                (group_num[..., None, None] + 1e-8)

            # Concat obs and nei feature
            f = torch.concat([f_obs, f_group], dim=-1)

        else:
            f_obs = self.te2(obs)
            f = f_obs
        
        if self.ranger_args.enable_exp_tran:
            f, _ = self.bb_lite_2(inputs=f, targets=f, training=training)

        # Compute expectation 
        f = self.exp(f)

        # Compute Conception and padding
        nei_trajs = nei * \
            (1 - group_mask[..., None, None]) + \
            group_mask[..., None, None] * INF
        
        if self.ranger_args.use_socialcircle:
            conception_circle = self.gp(obs, nei)
            f_social = self.tse_2(conception_circle)
        else:
            conception_circle = self.gp(obs, nei_trajs)

            f_social = self.tse(conception_circle)
            f_social = torch.repeat_interleave(f_social, torch.tensor(
                f_obs.shape[-2]).to(f_obs.device).to(torch.int32), dim=-2)
        
        if self.ranger_args.enable_per_tran:
            f_social, _ = self.bb_lite(inputs=f_social, targets=f_social, training=training)
        
        f_social = self.per(f_social)
        
        _f = f_social - f

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

        return Y


class Ranger(Structure):
    MODEL_TYPE = RangerModel
