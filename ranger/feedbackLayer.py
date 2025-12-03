import numpy as np
import torch

from qpid.model import layers


class ExpectationLayer(torch.nn.Module):

    def __init__(self, 
                 output_units: int,
                 hidden_units: int,
                 input_units: int,
                 use_activation: int | bool,
                 *args, **kwargs):

        super().__init__()

        self.output_units = output_units
        self.hidden_units = hidden_units
        self.input_units = input_units

        self.act = use_activation

        if self.act:
            self.fc1 = layers.Dense(self.input_units, self.hidden_units, torch.nn.ReLU)
            self.fc2 = layers.Dense(self.hidden_units, self.output_units, torch.nn.ReLU)
        else:
            self.fc1 = torch.nn.Linear(self.input_units, self.hidden_units)
            self.fc2 = torch.nn.Linear(self.hidden_units, self.output_units)

    def forward(self, feature: torch.tensor):

        return self.fc2(self.fc1(feature))

class PerceiveLayer(torch.nn.Module):

    def __init__(self, 
                 output_units: int,
                 hidden_units: int,
                 input_units: int,
                 use_activation: int | bool,
                 *args, **kwargs):

        super().__init__()

        self.output_units = output_units
        self.hidden_units = hidden_units
        self.input_units = input_units

        self.act = use_activation

        if self.act:
            self.fc1 = layers.Dense(self.input_units, self.hidden_units, torch.nn.ReLU)
            self.fc2 = layers.Dense(self.hidden_units, self.output_units, torch.nn.ReLU)
        else:
            self.fc1 = torch.nn.Linear(self.input_units, self.hidden_units)
            self.fc2 = torch.nn.Linear(self.hidden_units, self.output_units)

    def forward(self, feature: torch.tensor):

        return self.fc2(self.fc1(feature))
