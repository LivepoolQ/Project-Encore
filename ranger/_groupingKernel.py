import torch


class GroupingKernel(torch.nn.Module):
    def __init__(self, 
                 *args, **kwargs):
        super().__init__()

    def forward(self, ego_traj: torch.Tensor, nei_trajs: torch.Tensor):

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        

        return 
