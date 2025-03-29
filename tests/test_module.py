import numpy as np
import torch
from minitorch.tensor import Tensor
from minitorch.module import Linear, Softmax, Sequential, ReLU
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


def get_torch_minitorch_same_linear(in_features, out_features):
    model_torch = TLinear(in_features, out_features)
    model_minitorch = Linear(in_features, out_features)

    model_minitorch.weight.data = model_torch.weight.detach().numpy().transpose(1, 0)
    model_minitorch.bias.data = model_torch.bias.detach().numpy()[None, ...]

    return model_torch, model_minitorch


def test_mlp_classification_end_to_end():
    linear_1_torch, linear_1_minitorch = get_torch_minitorch_same_linear(728, 50)
    linear_2_torch, linear_2_minitorch = get_torch_minitorch_same_linear(50, 10)

    model_torch = torch.nn.Sequential(linear_1_torch, torch.nn.ReLU(), linear_2_torch)
    model_minitorch = Sequential(
        modules=[linear_1_minitorch, ReLU(), linear_2_minitorch]
    )

    batch_input = np.random.randn(32, 728)
    batch_labels = np.random.randint(0, 10, size=(32))

    batch_input_torch = torch.from_numpy(batch_input).float()
    batch_input_minitorch = Tensor(batch_input)

    batch_labels_torch = torch.from_numpy(batch_labels)
    batch_labels_minitorch = list(batch_labels)

    logits_torch = model_torch(batch_input_torch)
    logits_minitorch = model_minitorch(batch_input_minitorch)

    loss_torch = torch.nn.CrossEntropyLoss().forward(logits_torch, batch_labels_torch)
    loss_minitorch = cross_entropy_loss(logits_minitorch, batch_labels_minitorch)

    diff = np.abs((loss_torch.detach().numpy() - loss_minitorch.data)).sum()
    assert diff < 1e-6

    loss_torch.backward()
    loss_minitorch.backward()

    weight_grad_diff = np.abs(
        (
            linear_1_torch.weight.grad.detach().numpy().transpose(1, 0)
            - linear_1_minitorch.weight.grad.data
        )
    ).mean()
    assert weight_grad_diff < 1e-5
    bias_grad_diff = np.abs(
        (linear_1_torch.bias.grad.detach().numpy() - linear_1_minitorch.bias.grad.data)
    ).mean()
    assert bias_grad_diff < 1e-5


if __name__ == "__main__":
    test_mlp_classification_end_to_end()
