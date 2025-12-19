import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import RangerArgs
from ._groupLayer import GroupLayer, LongTermKernel, INF
from ._trajEncoding import TrajEncoding
from .egoPredictor import EgoPredictor
from .egoLoss import EgoLoss


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
            obs_steps=self.args.obs_frames//4,
            pred_steps=self.args.obs_frames//4,
            insights=self.ranger_args.insights_num,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim,
            use_ego_tran=self.ranger_args.use_ego_tran)

        if self.ranger_args.use_lite_egopredictor:
            self.ego_linear_predictor = layers.LinearLayerND(
                obs_frames=self.stage_len*2,
                pred_frames=self.stage_len*2,
                return_full_trajectory=False)
            
    def index_obs(self, n: int):
        """
        Decide range of stage.  

        :param n: n starts from 1.
        """
        return n * (self.stage_len) - 1 

    @property
    def stage_len(self):
        return self.args.obs_frames // self.N

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # mask neighbors
        nei_mask = (
            torch.sum(torch.abs(nei), dim=[-1, -2]) < (0.05 * INF)).to(dtype=torch.int32)

        if (t:=self.ranger_args.group_type) in [0, 1]:
            if t == 0:
                x_nei_pred_3_4 = None
                x_nei_pred_5_6 = None

            # -------------
            # Ego predictor
            # -------------
            elif t == 1:
                if training:
                    obs_2 = obs[..., self.ido(1):self.ido(2), :]
                    nei_2 = nei[..., self.ido(1):self.ido(2), :]
                    obs_3 = obs[..., self.ido(2):self.ido(3), :]
                    nei_3 = nei[..., self.ido(2):self.ido(3), :]
                    x_nei_pred_3, _ = self.ego_predictor(
                        ego_traj=obs_2,
                        nei_trajs=nei_2
                    )
                    x_nei_pred_4, _ = self.ego_predictor(
                        ego_traj=obs_3,
                        nei_trajs=nei_3
                    )
                    x_nei_pred_3_4 = torch.concat([
                        x_nei_pred_3,
                        x_nei_pred_4
                    ], dim=-2)
                    
                else:
                    x_nei_pred_3_4 = None

                if not self.ranger_args.use_lite_egopredictor:
                    # Use egopredictor predict stage4 -> stage5
                    x_nei_pred_5, x_ego_pred_5 = self.ego_predictor(
                        ego_traj=obs[..., self.ido(3):self.ido(4), :],
                        nei_trajs=nei[..., self.ido(3):self.ido(4), :]
                    )
                    x_nei_pred_6, _ = self.ego_predictor(
                        ego_traj=obs[..., self.ido(3):self.ido(4), :],
                        nei_trajs=torch.mean(x_nei_pred_5, dim=-3)
                    )

                    x_nei_pred_5_6 = torch.concat([x_nei_pred_5, x_nei_pred_6], dim=-2)
                    x_nei_pred_5_6 = torch.mean(x_nei_pred_5_6, dim=-3)

                # -------------
                # Linear variation
                # -------------
                else:
                    x_nei_pred_5_6 = self.ego_linear_predictor(nei[..., self.ido(2):self.ido(4), :])

                nei = torch.concat([nei[..., self.ido(2):self.ido(4), :], x_nei_pred_5_6], dim=-2)    
            
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

        return (Y,
                nei[..., self.index_obs(2):self.index_obs(4), :],
                x_nei_pred_3_4,
                x_nei_pred_5_6)


class Ranger(Structure):
    MODEL_TYPE = RangerModel

    def __init__(self, args=None,
                 manager=None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        self.ranger_args = self.args.register_subargs(
            RangerArgs, 'ranger_args')
        if ((not self.ranger_args.use_lite_egopredictor) 
            and (self.ranger_args.group_type == 1)):
            self.loss.set({l2: self.ranger_args.l2_loss_ratio, 
                           EgoLoss: self.ranger_args.ego_loss_ratio})
        else:
            self.loss.set({l2: 1.0})
