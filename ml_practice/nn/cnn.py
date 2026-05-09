"""简易 CNN (torch).
PREFERRED: torch

架构示例 (MNIST 风格):
  Conv(1,16,3,pad=1) -> ReLU -> MaxPool(2)
  Conv(16,32,3,pad=1) -> ReLU -> MaxPool(2)
  Flatten -> Linear(32*7*7, 10)
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO: 定义上述层
        raise NotImplementedError

    def forward(self, x):
        # x: (B, 1, 28, 28)
        # TODO
        raise NotImplementedError
