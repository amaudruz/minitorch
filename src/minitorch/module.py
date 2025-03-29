import numpy as np
from minitorch.tensor import Tensor
from minitorch.parameter import Parameter
from typing import Iterator


class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, "Module"] = {}

    def register_parameter(self, param: Parameter, name: str) -> None:
        self._parameters[name] = param

    def register_module(self, module: "Module", name: str) -> None:
        self._modules[name] = module

    @property
    def parameters(self) -> Iterator[tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for param_name, param in module.parameters:
                yield f"{name}.{param_name}", param

    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplemented


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(
            np.random.uniform(
                -1 / np.sqrt(in_features),
                1 / np.sqrt(in_features),
                size=(in_features, out_features),
            )
        )
        self.bias = Parameter(
            np.random.uniform(
                -1 / np.sqrt(in_features),
                1 / np.sqrt(in_features),
                size=(1, out_features),
            )
        )

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


class Softmax(Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, input: Tensor) -> tuple[Tensor, Tensor]:
        logits = input.exp()
        sum_exp = logits.sum(dim=self.dim)
        normalized_logits = logits / sum_exp
        return input, normalized_logits
