import numpy as np
import torch
from minitorch.tensor import Tensor
from minitorch.module import Linear
from torch.nn import Linear as TLinear


def test_linear():
    linear_torch = TLinear(10, 1)
    linear_minitorch = Linear(10, 1)

    linear_minitorch.weight.data = linear_torch.weight.detach().numpy().transpose(1, 0)
    linear_minitorch.bias.data = linear_torch.bias.detach().numpy()[..., None]

    input = np.random.randn(10, 10)
    result_torch = linear_torch.forward(torch.from_numpy(input).float()).sum()
    result_minitorch = linear_minitorch(input=Tensor(input)).sum()

    diff = np.abs((result_torch.detach().numpy() - result_minitorch.data)).sum()
    assert diff < 1e-6

    result_torch.backward()
    result_minitorch.backward()
    weight_grad_diff = np.abs(
        (
            linear_torch.weight.grad.detach().numpy().transpose(1, 0)
            - linear_minitorch.weight.grad.data
        )
    ).sum()
    assert weight_grad_diff < 1e-5
    bias_grad_diff = np.abs(
        (linear_torch.bias.grad.detach().numpy() - linear_minitorch.bias.grad.data)
    ).sum()
    assert bias_grad_diff < 1e-5
