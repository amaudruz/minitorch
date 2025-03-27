import numpy as np
from minitorch.tensor import Tensor
import torch


def test_operation_matmul():

    left = np.random.randn(10, 5)
    right = np.random.randn(5, 8)
    activation_grad = np.random.randn(10, 8)

    left_minitensor = Tensor(left)
    right_minitensor = Tensor(right)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor @ right_minitensor
    result_tensor = left_tensor @ right_tensor

    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.grad = activation_grad
    result_minitensor.backward()

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_add():

    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left)
    right_minitensor = Tensor(right)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor + right_minitensor
    result_tensor = left_tensor + right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.grad = activation_grad
    result_minitensor.backward()

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_sub():

    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left)
    right_minitensor = Tensor(right)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor - right_minitensor
    result_tensor = left_tensor - right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.grad = activation_grad
    result_minitensor.backward()

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_sum():

    data = np.random.randn(10, 5)

    data_minitensor = Tensor(data)

    data_tensor = torch.from_numpy(data).requires_grad_()

    for dim in [None, 0, 1]:

        data_tensor.grad = None
        summed_minitensor = data_minitensor.sum(dim)
        summed_tensor = data_tensor.sum(dim)
        diff = np.abs(
            (summed_minitensor.data.squeeze() - summed_tensor.detach().numpy())
        ).sum()
        assert diff < 1e-10

        if dim is None:
            activation_grad = np.random.randn(1, 1)
        elif dim == 0:
            activation_grad = np.random.randn(1, data.shape[1])
        elif dim == 1:
            activation_grad = np.random.randn(data.shape[0], 1)
        loss = (summed_tensor * torch.from_numpy(activation_grad).squeeze()).sum()
        loss.backward()

        summed_minitensor.grad = activation_grad
        summed_minitensor.backward()

        grad_diff = np.abs(
            data_minitensor.grad.squeeze() - data_tensor.grad.numpy()
        ).sum()
        assert grad_diff < 1e-10
