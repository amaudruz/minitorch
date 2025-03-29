import numpy as np
from minitorch.tensor import Tensor
import torch


def test_operation_matmul():
    left = np.random.randn(10, 5)
    right = np.random.randn(5, 8)
    activation_grad = np.random.randn(10, 8)

    left_minitensor = Tensor(left, retain_grad=True)
    right_minitensor = Tensor(right, retain_grad=True)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor @ right_minitensor
    result_tensor = left_tensor @ right_tensor

    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.backward(activation_grad)

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_mul():
    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left, retain_grad=True)
    right_minitensor = Tensor(right, retain_grad=True)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor * right_minitensor
    result_tensor = left_tensor * right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.backward(activation_grad)

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_div():
    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left, retain_grad=True)
    right_minitensor = Tensor(right, retain_grad=True)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor / right_minitensor
    result_tensor = left_tensor / right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.backward(activation_grad)

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_add():
    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left, retain_grad=True)
    right_minitensor = Tensor(right, retain_grad=True)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor + right_minitensor
    result_tensor = left_tensor + right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.backward(activation_grad)

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_sub():
    left = np.random.randn(10, 5)
    right = np.random.randn(1, 5)
    activation_grad = np.random.randn(10, 5)

    left_minitensor = Tensor(left, retain_grad=True)
    right_minitensor = Tensor(right, retain_grad=True)

    left_tensor = torch.from_numpy(left).requires_grad_()
    right_tensor = torch.from_numpy(right).requires_grad_()

    result_minitensor = left_minitensor - right_minitensor
    result_tensor = left_tensor - right_tensor
    diff = np.abs((result_minitensor.data - result_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (result_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    result_minitensor.backward(activation_grad)

    left_diff = np.abs(left_minitensor.grad - left_tensor.grad.numpy()).sum()
    right_diff = np.abs(right_minitensor.grad - right_tensor.grad.numpy()).sum()
    assert left_diff < 1e-10
    assert right_diff < 1e-10


def test_operation_mean():
    data = np.random.randn(10, 5)

    data_minitensor = Tensor(data, retain_grad=True)

    data_tensor = torch.from_numpy(data).requires_grad_()

    for dim in [None, 0, 1]:
        data_tensor.grad = None
        data_minitensor.grad = None
        mean_minitensor = data_minitensor.mean(dim)
        mean_tensor = data_tensor.mean(dim)
        mean_tensor.retain_grad()
        diff = np.abs(
            (mean_minitensor.data.squeeze() - mean_tensor.detach().numpy())
        ).sum()
        assert diff < 1e-10

        if dim is None:
            activation_grad = np.random.randn(1, 1)
        elif dim == 0:
            activation_grad = np.random.randn(1, data.shape[1])
        elif dim == 1:
            activation_grad = np.random.randn(data.shape[0], 1)
        loss = (mean_tensor * torch.from_numpy(activation_grad).squeeze()).sum()
        loss.backward()

        mean_minitensor.backward(activation_grad)

        grad_diff = np.abs(
            data_minitensor.grad.squeeze() - data_tensor.grad.numpy()
        ).sum()
        assert grad_diff < 1e-10


def test_operation_sum():
    data = np.random.randn(10, 5)

    data_minitensor = Tensor(data, retain_grad=True)

    data_tensor = torch.from_numpy(data).requires_grad_()

    for dim in [None, 0, 1]:
        data_tensor.grad = None
        data_minitensor.grad = None
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

        summed_minitensor.backward(activation_grad)

        grad_diff = np.abs(
            data_minitensor.grad.squeeze() - data_tensor.grad.numpy()
        ).sum()
        assert grad_diff < 1e-10


def test_operation_square():
    data = np.random.randn(10, 5)
    activation_grad = np.random.randn(10, 5)

    data_minitensor = Tensor(data, retain_grad=True)

    data_tensor = torch.from_numpy(data).requires_grad_()

    square_minitensor = data_minitensor.square()
    square_tensor = data_tensor.square()
    diff = np.abs(
        (square_minitensor.data.squeeze() - square_tensor.detach().numpy())
    ).sum()
    assert diff < 1e-10

    loss = (square_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    square_minitensor.backward(activation_grad)

    grad_diff = np.abs(data_minitensor.grad - data_tensor.grad.numpy()).sum()
    assert grad_diff < 1e-10


def test_operation_relu():
    data = np.random.randn(10, 5)
    activation_grad = np.random.randn(10, 5)

    data_minitensor = Tensor(data, retain_grad=True)

    data_tensor = torch.from_numpy(data).requires_grad_()

    relu_minitensor = data_minitensor.relu()
    relu_tensor = torch.nn.ReLU().forward(data_tensor)
    diff = np.abs((relu_minitensor.data.squeeze() - relu_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (relu_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    relu_minitensor.backward(activation_grad)

    grad_diff = np.abs(data_minitensor.grad - data_tensor.grad.numpy()).sum()
    assert grad_diff < 1e-10


def test_operation_two():
    data = np.random.randn(10, 5)
    activation_grad = np.random.randn(10, 5)

    data_minitensor = Tensor(data, retain_grad=True)

    data_tensor = torch.from_numpy(data).requires_grad_()

    op_minitensor = data_minitensor + data_minitensor
    op_tensor = data_tensor + data_tensor
    diff = np.abs((op_minitensor.data.squeeze() - op_tensor.detach().numpy())).sum()
    assert diff < 1e-10

    loss = (op_tensor * torch.from_numpy(activation_grad)).sum()
    loss.backward()

    op_minitensor.backward(activation_grad)

    grad_diff = np.abs(data_minitensor.grad - data_tensor.grad.numpy()).sum()
    assert grad_diff < 1e-10


if __name__ == "__main__":
    test_operation_div()
