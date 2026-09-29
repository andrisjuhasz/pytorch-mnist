import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

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
        [train_size,validation_size],
        generator=random_gen
        )

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_data,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)


loss_fn = nn.CrossEntropyLoss()

def train_one_epoch(model, optimizer, loader):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        avg_loss = loss_fn(logits, y) #average loss for the batch (reduction = "mean" by default)

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        total_loss += avg_loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples

def evaluate(model, loader):
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

            total_loss += avg_loss.item() * x.size(0)

            predictions = logits.argmax(dim=1)

            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

    return total_loss/total_samples, total_correct/total_samples

def training_validation_run(learning_rate, momentum):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        momentum=momentum,
        lr=learning_rate
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
            "validation_accuracies": validation_accuracies
            }

    return model, history

def print_history(history):
    for epoch in range(n_epochs):
        print(
            f"Epoch {epoch + 1:02d} | ", end=" ")
        for metric in history:
            print(f"{metric}: {history[metric][epoch]:.4f} |", end=" ")
        print()

def plot_metric_histories(histories, metric):
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

def main():
    lr_mom_experiments = [
        (0.4, 0.0),
        (0.15, 0.5),
        (0.03, 0.9),
    ]

    histories = {}
    models = {}

    for learning_rate, momentum in lr_mom_experiments:
        models[(learning_rate, momentum)], histories[(learning_rate, momentum)] = \
        training_validation_run(
            learning_rate,
            momentum
        )

    for learning_rate, momentum in lr_mom_experiments:
        print_history(histories[(learning_rate, momentum)])

    metrics = next(iter(histories.values())).keys()
    for metric in metrics:
        plot_metric_histories(histories, metric)
    plt.show()


    final_learning_rate, final_momentum = lr_mom_experiments[0]
    final_model = models[(final_learning_rate, final_momentum)]
    test_loss, test_accuracy = evaluate(final_model, test_loader)

    print(
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: {test_accuracy * 100:.2f}%"
    )

if __name__ == "__main__":
    main()

