import numpy as np
import torch
from minitorch.tensor import Tensor
from minitorch.module import Linear, Softmax
from minitorch.utils import cross_entropy_loss
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


def test_softmax():
    softmax_minitorch = Softmax()
    softmax_torch = torch.nn.Softmax(dim=0)

    input = np.random.randn(10, 1)
    torch_input = torch.from_numpy(input).float().requires_grad_()
    minitorch_input = Tensor(input, retain_grad=True)

    result_torch = softmax_torch.forward(torch_input)
    _, result_minitorch = softmax_minitorch(minitorch_input)

    diff = np.abs((result_torch.detach().numpy() - result_minitorch.data)).sum()
    assert diff < 1e-6

    result_torch.sum().backward()
    result_minitorch.sum().backward()

    grad_diff = np.abs((torch_input.grad.detach().numpy() - minitorch_input.grad)).sum()
    assert grad_diff < 1e-6


def test_cross_entropy():
    loss_torch = torch.nn.CrossEntropyLoss()

    logits = np.random.randn(10, 5)
    logits_torch = torch.from_numpy(logits).float().requires_grad_()
    logits_minitorch = Tensor(logits, retain_grad=True)

    target = np.random.randint(0, 5, size=(10))
    target_torch = torch.from_numpy(target)
    target_list = target.tolist()

    torch_loss = loss_torch(logits_torch, target_torch)
    minitorch_loss = cross_entropy_loss(logits=logits_minitorch, labels=target_list)

    diff = np.abs((torch_loss.detach().numpy() - minitorch_loss.data)).sum()
    assert diff < 1e-6

    torch_loss.backward()
    minitorch_loss.backward()

    grad_diff = np.abs(
        (logits_torch.grad.detach().numpy() - logits_minitorch.grad)
    ).sum()
    assert grad_diff < 1e-6


if __name__ == "__main__":
    test_cross_entropy()
