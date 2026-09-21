import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

train_val_ratio = 0.9

batch_size = 64
learning_rate = 0.01
momentum = 0.9
epochs = 10


transform = transforms.Compose([
    transforms.ToTensor()
])


full_train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_size = int(train_val_ratio * len(full_train_data))
val_size = len(full_train_data) - train_size

random_gen = torch.Generator().manual_seed(42)

train_data, validation_data = random_split(
        full_train_data,
        [train_size,val_size],
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


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(
    model.parameters(),
    momentum=momentum,
    lr=learning_rate
)


def train_one_epoch():
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y) #average loss for the batch (reduction = "mean" by default)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


for epoch in range(epochs):
    train_loss = train_one_epoch()
    test_accuracy = evaluate()

    print(
        f"Epoch {epoch + 1:02d} | "
        f"loss: {train_loss:.4f} | "
        f"test accuracy: {test_accuracy * 100:.2f}%"
    )
