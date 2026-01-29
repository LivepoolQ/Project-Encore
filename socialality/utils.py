"""
@Author: Conghao Wong
@Date: 2025-12-09 15:50:31
@LastEditors: Ziqian Zou
@LastEditTime: 2026-01-28 19:44:18
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import matplotlib.pyplot as plt
import torch

from qpid.model import layers


class KernelLayer(torch.nn.Module):
    """
    Kernel Layer
    ---
    The 3-layer MLP to compute reverberation kernels.
    `ReLU` is used in the first two layers, while `tanh` is used in the
    output layer.
    """

    def __init__(self, input_units: int,
                 hidden_units: int,
                 output_units: int,
                 *args, **kwargs) -> None:

        super().__init__()

        self.l1 = layers.Dense(input_units, hidden_units, torch.nn.ReLU)
        self.l2 = layers.Dense(hidden_units, hidden_units, torch.nn.ReLU)
        self.l3 = layers.Dense(hidden_units, output_units, torch.nn.Tanh)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l2(self.l1(f)))
    

def tensor_size_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**2


def print_variable_summary(locals):
    items = []
    for k, v in locals.items():
        if torch.is_tensor(v):
            items.append((tensor_size_mb(v), k, v))

    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    for size, k, v in items_sorted:
        print(
            f"{k:>25}  {str(tuple(v.shape)):>20}  {size:10.2f} MB  {v.dtype}  {v.device}")

    # summary
    total = sum(s for s, _, _ in items_sorted)
    n = len(items_sorted)
    max_item = items_sorted[0] if n else None

    print("-" * 100)
    if n and max_item:
        print(
            f"SUMMARY: {n} tensors, total {total:.2f} MB, largest {max_item[1]} {tuple(max_item[2].shape)} {max_item[0]:.2f} MB")
    else:
        print("SUMMARY: 0 tensors")

def repeat(input: torch.Tensor, repeats: int, dim: int):
    shape = input.shape
    d = dim % len(shape)
    x = input.unsqueeze(d+1)
    x = x.expand(*shape[:d+1], repeats, *shape[d+1:])
    x = x.flatten(d, d+1)
    return x


class Gate(torch.nn.Module):

    def __init__(self, 
                 gate_value: float = -0.9,
                 *args, **kwargs) -> None:
        super().__init__()
        self.gate = gate_value

    def forward(self, x, *args, **kwargs):
        return torch.maximum(x, torch.ones_like(x) * self.gate)
    
    
def vis_socialality(anchors:torch.Tensor):

    plt.figure(figsize=(4, 4))

    plt.scatter(
        anchors[:, 0],
        anchors[:, 1],
        s=20,
        alpha=0.3
    )
    plt.xlabel(r'$\tau^{a}$')
    plt.ylabel(r'$\tau^{b}$')
    plt.title('Socialality Anchors')
    plt.axis('equal')

    import seaborn as sns
    sns.kdeplot(x=anchors.numpy().T[0], y=anchors.numpy().T[1], fill=True, alpha=0.5)

    plt.show()

    return anchors
    
