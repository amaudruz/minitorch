import numpy as np
from minitorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data, retain_grad=True)
