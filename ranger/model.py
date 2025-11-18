import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model
from qpid.training import Structure

from .__args import RangerArgs


class RangerModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs_traj = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        pred = torch.repeat_interleave(
            obs_traj[:, 0:1, :], self.args.pred_frames, dim=1)

        return pred


class Ranger(Structure):
    is_trainable = False
    MODEL_TYPE = RangerModel
