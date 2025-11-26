"""
@Author: Ziqian Zou
@Date: 2025-11-25 11:40:32
@LastEditors: Ziqian Zou
@LastEditTime: 2025-11-25 11:41:59
@Description: file content
@Github: https://github.com/LivepoolQ
@Copyright 2025 Ziqian Zou, All Rights Reserved.
"""
import torch


class TrajEncoding(torch.nn.Module):
    """

    """
    def __init__(self,
                 output_units: int,
                 input_units: int = 2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_units = output_units
        self.input_units = input_units

        self.fc1 = torch.nn.Linear(input_units, output_units)
        self.ac1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(output_units, output_units)
        self.ac2 = torch.nn.Tanh()

    def forward(self, trajs: torch.Tensor) -> torch.Tensor:

        f = self.fc1(trajs)
        f = self.ac1(f)
        f = self.fc2(f)
        trajs_enc = self.ac2(f)

        return trajs_enc