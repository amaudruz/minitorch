import numpy as np


class Tensor:

    def __init__(self, data: np.ndarray, operator=None) -> None:
        self.data = data
        self.operator = operator
        self.grad: np.ndarray | None = None

    def backward(self) -> None:
        assert self.grad is not None
        if self.operator:
            self.operator.backward(self.grad)

    @property
    def shape(self):
        return self.data.shape

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import MatMul

        return MatMul().forward(self, other)

    def __add__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import TwoDimenstionalAdd

        return TwoDimenstionalAdd().forward(self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        from minitorch.operations import TwoDimenstionalSub

        return TwoDimenstionalSub().forward(self, other)

    def sum(self, dim: int | None = None) -> "Tensor":
        from minitorch.operations import Sum

        return Sum().forward(self, dim)
