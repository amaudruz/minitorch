import numpy as np
from torch import relu


class Tensor:
    def __init__(
        self, data: np.ndarray, operator=None, retain_grad: bool = False
    ) -> None:
        self.data = data
        self.operator = operator
        self.grad: np.ndarray | None = None
        self.retain_grad = retain_grad

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            assert all(dim_size == 1 for dim_size in self.shape) and (self.grad is None)
            assert self.operator
            return self.operator.backward(np.ones((1, 1)))
        if self.retain_grad:
            self.grad = grad if self.grad is None else self.grad + grad
        if self.operator:
            self.operator.backward(grad)

    @property
    def shape(self):
        return self.data.shape

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import MatMul

        return MatMul().forward(self, other)

    def __add__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import TwoDimensionalAdd

        return TwoDimensionalAdd().forward(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import TwoDimensionalSub

        return TwoDimensionalSub().forward(self, other)

    def sum(self, dim: int | None = None) -> "Tensor":
        from minitorch.operations import Sum

        return Sum().forward(self, dim)

    def mean(self, dim: int | None = None) -> "Tensor":
        from minitorch.operations import Mean

        return Mean().forward(self, dim)

    def square(self) -> "Tensor":
        from minitorch.operations import Square

        return Square().forward(self)

    def relu(self) -> "Tensor":
        from minitorch.operations import ReLU

        return ReLU().forward(self)
