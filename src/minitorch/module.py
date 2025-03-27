import numpy as np
from minitorch.tensor import Tensor


class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}

    def register_parameter(self, param: Tensor, name: str) -> None:
        self._parameters[name] = param


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.random.randn(1, out_features))

        self.register_parameter(self.weight, "weight")
        self.register_parameter(self.bias, "bias")

    def __call__(self, input: Tensor) -> Tensor:
        return (input @ self.weight) + self.bias
