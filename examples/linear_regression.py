import numpy as np
from minitorch.module import Linear
from tqdm.auto import tqdm
from minitorch.tensor import Tensor

if __name__ == "__main__":

    n_samples = 400
    n_dims = 10
    inputs_dataset = np.random.randn(n_samples, n_dims)

    ground_truth_weights = np.random.randn(10, 1)
    ground_truth_bias = np.random.randn(1, 1)

    std = 0.8
    noise = np.random.randn(n_samples) * std
    labels = inputs_dataset @ ground_truth_weights + ground_truth_bias

    model = Linear(n_dims, 1)

    n_epochs = 10
    batch_size = 32
    learning_rate = 1e-2

    for epoch in range(n_epochs):
        batch_losses = []
        print(f"EPOCH: {epoch + 1}")
        for batch_index in tqdm(list(range(0, n_samples, batch_size)), leave=False):

            batch_input = Tensor(inputs_dataset[batch_index : batch_index + batch_size])
            batch_labels = Tensor(labels[batch_index : batch_index + batch_size])

            batch_predictions = model(input=batch_input)

            loss = (batch_predictions - batch_labels).square().mean()
            loss.backward()

            for param in model._parameters.values():
                param.data = param.data - (learning_rate * param.grad)

            batch_losses.append(loss.data)
        print(np.mean(batch_losses))
