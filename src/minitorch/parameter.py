import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: np.ndarray | None = None
