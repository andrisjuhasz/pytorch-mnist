from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models.fully_connected import make_fully_connected
from models.cnn import make_cnn

device = "cuda" if torch.cuda.is_available() else "cpu"

train_validation_ratio = 0.9
batch_size = 64
n_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor()
])

full_train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_size = int(train_validation_ratio * len(full_train_data))
validation_size = len(full_train_data) - train_size

random_gen = torch.Generator().manual_seed(42)

train_data, validation_data = random_split(
    full_train_data,
    [train_size, validation_size],
    generator=random_gen,
    )

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
)

loss_fn = nn.CrossEntropyLoss()


def train_one_epoch(model, optimizer, loader):
    """Train the model for one epoch and return the average loss."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        avg_loss = loss_fn(logits, y)

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        # Weight the batch mean by batch size to account for a potentially
        # smaller final batch.
        total_loss += avg_loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


def evaluate(model, loader):
    """Return the average loss and accuracy on a dataset."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            avg_loss = loss_fn(logits, y)

            # Weight the batch mean by batch size to account for a potentially
            # smaller final batch.
            total_loss += avg_loss.item() * x.size(0)

            predictions = logits.argmax(dim=1)

            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def training_validation_run(model, learning_rate, momentum):
    """Train and validate a model using the given SGD hyperparameters."""
    # Reset the RNG so experiments use the same model initialization
    # and mini-batch ordering.

    optimizer = torch.optim.SGD(
        model.parameters(),
        momentum=momentum,
        lr=learning_rate,
    )

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader)
        validation_loss, validation_accuracy = evaluate(model, validation_loader)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    history = {
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
        }

    return history


def print_history(history):
    """Print the recorded training and validation metrics for each epoch."""
    for epoch in range(n_epochs):
        print(
            f"Epoch {epoch + 1:02d} | ", end=" ")
        for metric in history:
            print(f"{metric}: {history[metric][epoch]:.4f} |", end=" ")
        print()

def plot_metric_histories(histories, metric):
    """Save a metric plot for each hyperparameter configuration."""
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for (learning_rate, momentum), history in histories.items():
        values = history[metric]
        epochs = range(1, len(values) + 1)

        plt.plot(
            epochs,
            values,
            label=f"lr={learning_rate}, m={momentum}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = output_dir / f"{metric}_{timestamp}.png"

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath

def main():
    lr_momentum_experiments = [
        (0.4, 0.0),
        (0.15, 0.5),
        (0.03, 0.9),
    ]

    histories = {}
    models = {}

    for learning_rate, momentum in lr_momentum_experiments:
        torch.manual_seed(42)
        model = make_cnn().to(device)
        history = training_validation_run(
                model,
                learning_rate,
                momentum,
                )
        models[(learning_rate, momentum)] = model
        histories[(learning_rate, momentum)] = history

    for learning_rate, momentum in lr_momentum_experiments:
        print_history(histories[(learning_rate, momentum)])

    metrics = next(iter(histories.values())).keys()
    for metric in metrics:
        plot_metric_histories(histories, metric)

    # Hyperparameters selected based on validation performance.
    final_learning_rate = 0.4
    final_momentum = 0.0

    final_model = models[(final_learning_rate, final_momentum)]

    # Evaluate the selected model on the test set only after model selection.
    test_loss, test_accuracy = evaluate(final_model, test_loader)

    print(
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )

if __name__ == "__main__":
    main()

