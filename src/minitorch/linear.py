import numpy as np
from minitorch.parameter import Parameter


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.a = Parameter(
            np.random.randn(
                in_features,
                out_features,
            )
        )
        self.b = Parameter(np.random.randn(out_features))

    def __call__(self, input: np.ndarray) -> np.ndarray:

        bs, in_features = input.shape

        assert (
            in_features == self.in_features
        ), f"Input array dim {in_features} is not equal to expected dim {self.in_features}"

        return (input @ self.a.data) + self.b.data

    def backward(self, backprop: np.ndarray) -> chr
