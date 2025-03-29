import numpy as np
from abc import ABC, abstractmethod
from minitorch.tensor import Tensor


class Operation(ABC):
    @abstractmethod
    def forward(self, *inputs) -> Tensor:
        pass

    @abstractmethod
    def backward(self, activation_grad: np.ndarray) -> None:
        pass


class MatMul(Operation):
    def __init__(self) -> None:
        self.left: Tensor | None = None
        self.right: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        left, right = inputs
        assert isinstance(left, Tensor) and len(left.data.shape) == 2
        assert isinstance(right, Tensor) and len(right.data.shape) == 2

        self.left, self.right = left, right
        return Tensor(self.left.data @ self.right.data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (self.left is not None) and (self.right is not None), (
            "Operation must be executed before calling backward on it"
        )

        self.right.backward(self.left.data.transpose() @ activation_grad)
        self.left.backward(activation_grad @ self.right.data.transpose())


class TwoDimensionalDiv(Operation):
    def __init__(self) -> None:
        self.left: Tensor | None = None
        self.right: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        left, right = inputs
        assert isinstance(left, Tensor) and len(left.data.shape) == 2
        assert isinstance(right, Tensor) and len(right.data.shape) == 2

        self.left, self.right = left, right
        return Tensor(self.left.data / self.right.data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (self.left is not None) and (self.right is not None), (
            "Operation must be executed before calling backward on it"
        )

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        right_grad = (
            self.left.data
            * activation_grad.copy()
            / -(self.right.data * self.right.data)
        )
        for dim in right_broadcasting_dims:
            right_grad = right_grad.sum(dim, keepdims=True)

        left_grad = activation_grad.copy() / self.right.data
        for dim in left_broadcasting_dims:
            left_grad = left_grad.sum(dim, keepdims=True)

        self.right.backward(right_grad)
        self.left.backward(left_grad)


class TwoDimensionalMul(Operation):
    def __init__(self) -> None:
        self.left: Tensor | None = None
        self.right: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        left, right = inputs
        assert isinstance(left, Tensor) and len(left.data.shape) == 2
        assert isinstance(right, Tensor) and len(right.data.shape) == 2

        self.left, self.right = left, right
        return Tensor(self.left.data * self.right.data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (self.left is not None) and (self.right is not None), (
            "Operation must be executed before calling backward on it"
        )

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        right_grad = self.left.data * activation_grad.copy()
        for dim in right_broadcasting_dims:
            right_grad = right_grad.sum(dim, keepdims=True)

        left_grad = self.right.data * activation_grad.copy()
        for dim in left_broadcasting_dims:
            left_grad = left_grad.sum(dim, keepdims=True)

        self.right.backward(right_grad)
        self.left.backward(left_grad)


class TwoDimensionalAdd(Operation):
    def __init__(self) -> None:
        self.left: Tensor | None = None
        self.right: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        left, right = inputs
        assert isinstance(left, Tensor) and len(left.data.shape) == 2
        assert isinstance(right, Tensor) and len(right.data.shape) == 2

        self.left, self.right = left, right
        return Tensor(self.left.data + self.right.data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (self.left is not None) and (self.right is not None), (
            "Operation must be executed before calling backward on it"
        )

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        right_grad = activation_grad.copy()
        for dim in right_broadcasting_dims:
            right_grad = right_grad.sum(dim, keepdims=True)

        left_grad = activation_grad.copy()
        for dim in left_broadcasting_dims:
            left_grad = left_grad.sum(dim, keepdims=True)

        self.right.backward(right_grad)
        self.left.backward(left_grad)


class TwoDimensionalSub(Operation):
    def __init__(self) -> None:
        self.left: Tensor | None = None
        self.right: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        left, right = inputs
        assert isinstance(left, Tensor) and len(left.data.shape) == 2
        assert isinstance(right, Tensor) and len(right.data.shape) == 2

        self.left, self.right = left, right
        return Tensor(self.left.data - self.right.data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (self.left is not None) and (self.right is not None), (
            "Operation must be executed before calling backward on it"
        )

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        right_grad = activation_grad.copy()
        for dim in right_broadcasting_dims:
            right_grad = -right_grad.sum(dim, keepdims=True)

        left_grad = activation_grad.copy()
        for dim in left_broadcasting_dims:
            left_grad = left_grad.sum(dim, keepdims=True)

        self.right.backward(right_grad)
        self.left.backward(left_grad)


class Mean(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        tensor, dim = inputs
        assert (dim == 1) or (dim == 0) or (dim is None)
        assert isinstance(tensor, Tensor)
        self.tensor, self.dim = tensor, dim
        mean_data = (
            self.tensor.data.mean(self.dim, keepdims=True)
            if self.dim is not None
            else self.tensor.data.mean(keepdims=True)
        )
        return Tensor(mean_data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None, (
            "Operation must be executed before calling backward on it"
        )
        assert len(activation_grad.shape) == 2

        scale = (
            (self.tensor.data.size) if self.dim is None else self.tensor.shape[self.dim]
        )
        grad = np.ones_like(self.tensor.data) * activation_grad / scale
        self.tensor.backward(grad)


class Sum(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        tensor, dim = inputs
        assert (dim == 1) or (dim == 0) or (dim is None)
        assert isinstance(tensor, Tensor)
        self.tensor, self.dim = tensor, dim
        summed_data = (
            self.tensor.data.sum(self.dim, keepdims=True)
            if self.dim is not None
            else self.tensor.data.sum(keepdims=True)
        )
        return Tensor(summed_data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None, (
            "Operation must be executed before calling backward on it"
        )
        assert len(activation_grad.shape) == 2

        grad = np.ones_like(self.tensor.data) * activation_grad
        self.tensor.backward(grad)


class Square(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        (tensor,) = inputs
        assert isinstance(tensor, Tensor)
        self.tensor = tensor

        return Tensor(np.square(self.tensor.data), self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None
        grad = 2 * self.tensor.data * activation_grad
        self.tensor.backward(grad)


class Exp(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        (tensor,) = inputs
        assert isinstance(tensor, Tensor)
        self.tensor = tensor
        res = np.exp(self.tensor.data)

        return Tensor(np.exp(self.tensor.data), self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None
        grad = np.exp(self.tensor.data) * activation_grad
        self.tensor.backward(grad)


class Log(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        (tensor,) = inputs
        assert isinstance(tensor, Tensor)
        self.tensor = tensor

        return Tensor(np.log(self.tensor.data), self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None
        grad = activation_grad / (self.tensor.data)
        self.tensor.backward(grad)


class ReLU(Operation):
    def __init__(self) -> None:
        self.tensor: Tensor | None = None

    def forward(self, *inputs) -> Tensor:
        (tensor,) = inputs
        assert isinstance(tensor, Tensor)
        self.tensor = tensor

        return Tensor(self.tensor.data * (self.tensor.data > 0), self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert self.tensor is not None
        grad = (self.tensor.data > 0) * activation_grad
        self.tensor.backward(grad)
