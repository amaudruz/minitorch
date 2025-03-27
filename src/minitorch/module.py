import numpy as np
from minitorch.tensor import Tensor
from typing import Iterator


class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, "Module"] = {}

    def register_parameter(self, param: Tensor, name: str) -> None:
        self._parameters[name] = param

    def register_module(self, module: "Module", name: str) -> None:
        self._modules[name] = module

    @property
    def parameters(self) -> Iterator[tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for name, param in module.parameters:
                yield f"{module}.{name}", param

    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplemented


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features))
        self.bias = Tensor(np.random.randn(1, out_features))

        self.register_parameter(self.weight, "weight")
        self.register_parameter(self.bias, "bias")

    def __call__(self, input: Tensor) -> Tensor:
        return (input @ self.weight) + self.bias


class ReLU(Module):

    def __call__(self, input: Tensor) -> Tensor:
        return input.relu()


class Sequential(Module):
    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules
        for i, module in enumerate(self.modules):
            self.register_module(module, f"Sequential.{i}")

    def __call__(self, input: Tensor) -> Tensor:
        output = input
        for module in self.modules:
            output = module(output)
        return output
