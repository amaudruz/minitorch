import numpy as np
from minitorch.module import Linear, ReLU, Sequential
from minitorch.tensor import Tensor
from tqdm.auto import tqdm
import torch.nn as nn
import torch

if __name__ == "__main__":

    n_samples = 400
    n_dims = 10
    intermediate_features = 5
    inputs_dataset = np.random.randn(n_samples, n_dims)

    ground_truth_model = nn.Sequential(
        nn.Linear(n_dims, intermediate_features),
        nn.ReLU(),
        nn.Linear(intermediate_features, 1),
    )
    minitorch_model = Sequential(
        modules=[
            Linear(n_dims, intermediate_features),
            ReLU(),
            Linear(intermediate_features, 1),
        ]
    )

    labels = (
        ground_truth_model.forward(torch.from_numpy(inputs_dataset).float())
        .detach()
        .numpy()
    )

    n_epochs = 10
    batch_size = 32
    learning_rate = 1e-2

    for epoch in range(n_epochs):
        batch_losses = []
        print(f"EPOCH: {epoch + 1}")
        for batch_index in tqdm(list(range(0, n_samples, batch_size)), leave=False):

            batch_input = Tensor(inputs_dataset[batch_index : batch_index + batch_size])
            batch_labels = Tensor(labels[batch_index : batch_index + batch_size])

            batch_predictions = minitorch_model(input=batch_input)

            loss = (batch_predictions - batch_labels).square().mean()
            loss.backward()

            for name, param in minitorch_model.parameters:
                param.data = param.data - (learning_rate * param.grad)
                param.grad = None

            batch_losses.append(loss.data)
        print(np.mean(batch_losses))
