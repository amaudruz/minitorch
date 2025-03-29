import os
import numpy as np
from minitorch.module import Linear, ReLU, Sequential, Softmax
from minitorch.tensor import Tensor
from minitorch.utils import cross_entropy_loss
from tqdm.auto import tqdm
from datasets import load_dataset
from pathlib import Path

train_ds_path = Path("train_ds.npk")
train_labels_path = Path("train_labels.npk")
valid_ds_path = Path("valid_ds.npk")
valid_labels_path = Path("valid_labels.npk")

if train_ds_path.exists():
    train_ds = np.load(train_ds_path.open("rb"))
    train_labels = np.load(train_labels_path.open("rb"))
    valid_ds = np.load(valid_ds_path.open("rb"))
    valid_labels = np.load(valid_labels_path.open("rb"))
else:
    print("Loading dataset")
    ds = load_dataset("ylecun/mnist")

    train_ds = np.stack([np.array(item["image"]) for item in ds["train"]]) / 255
    train_labels = np.array([item["label"] for item in ds["train"]])

    valid_ds = np.stack([np.array(item["image"]) for item in ds["test"]]) / 255
    valid_labels = np.array([item["label"] for item in ds["test"]])

    print("Caching datasets")
    np.save(train_ds_path.open("wb"), train_ds)
    np.save(train_labels_path.open("wb"), train_labels)
    np.save(valid_ds_path.open("wb"), valid_ds)
    np.save(valid_labels_path.open("wb"), valid_labels)


flat_train_ds = train_ds.reshape(train_ds.shape[0], -1)
flat_valid_ds = valid_ds.reshape(valid_ds.shape[0], -1)

image_width, image_height = 28, 28
model = Sequential(
    modules=[
        Linear(image_width * image_height, 50),
        ReLU(),
        Linear(50, 20),
        ReLU(),
        Linear(20, 10),
    ]
)

n_epochs = 15
batch_size = 128
learning_rate = 1e-1

print("Running training")
for epoch in range(n_epochs):
    print(f"EPOCH: {epoch + 1}")

    batch_losses = []
    batch_accuracies = []
    for batch_index in tqdm(
        list(range(0, len(flat_train_ds), batch_size)), leave=False
    ):

        batch_input = Tensor(flat_train_ds[batch_index : batch_index + batch_size])
        batch_labels = train_labels[batch_index : batch_index + batch_size]

        logits = model(input=batch_input)
        loss = cross_entropy_loss(logits, list(batch_labels))
        loss.backward()

        for name, param in model.parameters:
            param.data = param.data - (learning_rate * param.grad)
            param.grad = None

        predictions = np.argmax(logits.data, axis=1)
        batch_accuracy = (predictions == np.array(batch_labels)).mean()

        batch_losses.append(loss.data)
        batch_accuracies.append(batch_accuracy)

    print(
        f"Mean train loss: {np.mean(batch_losses)}, Mean train accurarcy: {np.mean(batch_accuracies)}"
    )

    batch_losses = []
    batch_accuracies = []
    for batch_index in tqdm(
        list(range(0, len(flat_valid_ds), batch_size)), leave=False
    ):

        batch_input = Tensor(flat_valid_ds[batch_index : batch_index + batch_size])
        batch_labels = valid_labels[batch_index : batch_index + batch_size]

        logits = model(input=batch_input)

        loss = cross_entropy_loss(logits, list(batch_labels))

        predictions = np.argmax(logits.data, axis=1)
        batch_accuracy = (predictions == np.array(batch_labels)).mean()

        batch_losses.append(loss.data)
        batch_accuracies.append(batch_accuracy)

    print(
        f"Mean valid loss: {np.mean(batch_losses)}, Mean valid accurarcy: {np.mean(batch_accuracies)}"
    )
