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
        assert (self.left is not None) and (
            self.right is not None
        ), "Operation must be executed before calling backward on it"

        self.right.grad = self.left.data.transpose() @ activation_grad
        self.left.grad = activation_grad @ self.right.data.transpose()

        self.right.backward()
        self.left.backward()

        self.a = None
        self.b = None


class TwoDimenstionalAdd(Operation):

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
        assert (self.left is not None) and (
            self.right is not None
        ), "Operation must be executed before calling backward on it"

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        self.right.grad = activation_grad.copy()
        for dim in right_broadcasting_dims:
            self.right.grad = self.right.grad.sum(dim, keepdims=True)

        self.left.grad = activation_grad.copy()
        for dim in left_broadcasting_dims:
            self.left.grad = self.left.grad.sum(dim, keepdims=True)

        self.right.backward()
        self.left.backward()

        self.a = None
        self.b = None


class TwoDimenstionalSub(Operation):

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
        assert (self.left is not None) and (
            self.right is not None
        ), "Operation must be executed before calling backward on it"

        def _get_2d_broadcasting_dims(array_2d: np.ndarray) -> list[int]:
            return [i for i in range(2) if array_2d.shape[i] == 1]

        left_broadcasting_dims = _get_2d_broadcasting_dims(self.left.data)
        right_broadcasting_dims = _get_2d_broadcasting_dims(self.right.data)

        self.right.grad = activation_grad.copy()
        for dim in right_broadcasting_dims:
            self.right.grad = -self.right.grad.sum(dim, keepdims=True)

        self.left.grad = activation_grad.copy()
        for dim in left_broadcasting_dims:
            self.left.grad = self.left.grad.sum(dim, keepdims=True)

        self.right.backward()
        self.left.backward()

        self.a = None
        self.b = None


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
            else self.tensor.data.sum()
        )
        return Tensor(summed_data, self)

    def backward(self, activation_grad: np.ndarray) -> None:
        assert (
            self.tensor is not None
        ), "Operation must be executed before calling backward on it"
        assert len(activation_grad.shape) == 2

        self.tensor.grad = np.ones_like(self.tensor.data) * activation_grad
        self.tensor.backward()

        self.tensor = None
