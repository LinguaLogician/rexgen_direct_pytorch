# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: nn.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:12
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: nn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class Linear(nn.Module):
    """Linear layer with custom initialization."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with custom scheme."""
        stddev = min(1.0 / math.sqrt(self.linear.in_features), 0.1)
        nn.init.normal_(self.linear.weight, std=stddev)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearND(nn.Module):
    """Linear layer for N-dimensional input."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with custom scheme."""
        stddev = min(1.0 / math.sqrt(self.input_dim), 0.1)
        nn.init.normal_(self.weight, std=stddev)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to N-dimensional input."""
        # Get original shape
        original_shape = x.shape
        ndim = x.dim()

        # Flatten all but last dimension
        x_flat = x.reshape(-1, self.input_dim)

        # Apply linear transformation
        output = F.linear(x_flat, self.weight, self.bias)

        # Reshape to original dimensions with new last dimension
        new_shape = list(original_shape[:-1]) + [self.output_dim]
        return output.reshape(new_shape)


def linear(input_: torch.Tensor, output_size: int, init_bias: float = 0.0) -> torch.Tensor:
    """
    Functional version of linear layer (for compatibility).
    """
    input_dim = input_.shape[-1]

    # 创建权重和偏置
    stddev = min(1.0 / math.sqrt(input_dim), 0.1)
    weight = torch.randn(output_size, input_dim) * stddev

    if init_bias is not None:
        bias = torch.full((output_size,), init_bias, dtype=input_.dtype)
        return F.linear(input_, weight, bias)
    else:
        return F.linear(input_, weight)


def linearND(input_: torch.Tensor, output_size: int, init_bias: float = 0.0) -> torch.Tensor:
    """
    Functional version of linearND layer (for compatibility).
    """
    # 创建 LinearND 层
    input_dim = input_.shape[-1]
    layer = LinearND(input_dim, output_size, bias=(init_bias is not None))

    # 如果指定了 init_bias，设置偏置值
    if init_bias is not None and layer.bias is not None:
        nn.init.constant_(layer.bias, init_bias)

    return layer(input_)


if __name__ == "__main__":
    # Test the modules
    batch_size = 2
    seq_len = 10
    input_dim = 32
    output_dim = 64

    # Test Linear
    print("Testing Linear layer:")
    x = torch.randn(batch_size, seq_len, input_dim)
    linear_layer = Linear(input_dim, output_dim)
    output = linear_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test LinearND
    print("\nTesting LinearND layer:")
    x_nd = torch.randn(batch_size, seq_len, 5, input_dim)  # 4D input
    linear_nd_layer = LinearND(input_dim, output_dim)
    output_nd = linear_nd_layer(x_nd)
    print(f"Input shape: {x_nd.shape}")
    print(f"Output shape: {output_nd.shape}")

    # Test functional versions
    print("\nTesting functional versions:")
    output_func = linear(x, output_dim, scope="test")
    print(f"linear output shape: {output_func.shape}")

    output_nd_func = linearND(x_nd, output_dim, scope="test_nd")
    print(f"linearND output shape: {output_nd_func.shape}")
