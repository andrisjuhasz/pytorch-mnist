# MNIST Classification with PyTorch

Fully connected and convolutional neural networks for handwritten digit classification, with a shared training and validation workflow.

This educational project explores how PyTorch components fit together: tensors and autograd, model modules, datasets and loaders, optimizers, and evaluation. It also includes reproducible hyperparameter comparisons and a command-line interface.

## Models

Both models accept batches shaped `(B, 1, 28, 28)` and produce ten logits per image. Training uses cross-entropy loss and mini-batch SGD.

| Model | Architecture |
|---|---|
| Fully connected | Flatten → Linear(784, 128) → ReLU → Linear(128, 10) |
| CNN | Conv2d(1, 16, 3, padding=1) → ReLU → MaxPool2d(2) → Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool2d(2) → Flatten → Linear(1568, 128) → ReLU → Linear(128, 10) |

The CNN preserves spatial dimensions in each convolution and halves them in each pooling layer: 28 × 28 → 14 × 14 → 7 × 7. Each model factory creates a fresh `nn.Sequential` instance.

## Source files

| File | Responsibility |
|---|---|
| `models/fully_connected.py` | Fully connected model factory |
| `models/cnn.py` | CNN model factory |
| `train.py` | Command-line arguments, data setup, training, evaluation, metric histories, and plotting |

## Running the project

Requirements: Python 3, PyTorch, torchvision, and Matplotlib.

Install dependencies using the repository's requirements file:

```bash
python -m pip install -r requirements.txt
```

Run either architecture:

```bash
python train.py --model fully_connected --epochs 20
python train.py --model cnn --epochs 20
```

For a quick check, reduce the epoch count:

```bash
python train.py --model cnn --epochs 1
```

Each invocation runs all three SGD configurations; `--epochs` specifies the number of epochs **per configuration**.

| Option | Allowed values | Default |
|---|---|---|
| `--model` | `fully_connected`, `cnn` | `fully_connected` |
| `--epochs` | Positive integer | `10` |

```bash
python train.py --help
```

MNIST is downloaded to `data/` on the first run. The script uses CUDA when available and otherwise runs on CPU. Paths are relative to the working directory; run commands from the repository root.

## Training and evaluation

- MNIST's 60,000 training images are split into 54,000 training and 6,000 validation images using a dedicated generator seeded with `42`.
- `ToTensor()` converts images to floating-point tensors with pixel values in `[0, 1]`.
- Training uses batches of 64, shuffled each epoch.
- Each run compares `(learning rate, momentum)` pairs `(0.4, 0.0)`, `(0.15, 0.5)`, and `(0.03, 0.9)`.
- The PyTorch seed is reset to `42` before creating each model, giving matching initialization and training batch ordering across configurations of the same architecture in the same environment. Exact reproducibility across platforms or PyTorch versions is not guaranteed.
- Validation loss and accuracy are measured after every epoch, with gradients disabled.
- Training and validation histories are printed after the experiments. Three timestamped PNG plots are saved to `plots/`: training loss, validation loss, and validation accuracy.

**Current selection behavior:** test evaluation uses the final-epoch model with learning rate `0.4` and momentum `0.0`, for either architecture. This choice is hardcoded; the script does not yet select the best run or restore the best validation epoch. The held-out test set contains 10,000 images.

## Earlier fully connected baseline results

An earlier 20-epoch baseline experiment achieved **98.19% test accuracy** with test loss **0.0730**, using learning rate `0.4` and momentum `0.0`.

These are historical results, not a benchmark of the current script's default settings. No final CNN test result is reported here yet.

## PyTorch concepts explored

The project was built as a way to learn PyTorch beyond simply using a pre-written training loop. In particular, it provided practical experience with:

- **Tensors and autograd** — tensor operations, computation graphs, gradient tracking, `backward()`, and parameter gradients.
- **`nn.Module`** — PyTorch's model abstraction and the use of modules such as `nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear`, `nn.ReLU`, `nn.Flatten`, `nn.Sequential`, and `nn.CrossEntropyLoss`.
- **Datasets and `DataLoader`** — dataset access, train/validation splitting, batching, shuffling, and iteration over mini-batches.
- **Optimizers** — `torch.optim.Optimizer` and SGD, including the effects of learning rate and momentum.
- **Training and evaluation modes** — `model.train()`, `model.eval()`, and `torch.no_grad()`.
- **`torchvision`** — loading MNIST and applying input transformations with `transforms.Compose` and `transforms.ToTensor`.
- **Reproducibility** — explicit random generators and manual seeding for reproducible dataset splits and comparable experiments.
- **Validation and hyperparameter selection** — separating training, validation, and test data and using them for their distinct purposes.

## Next steps

- Automate hyperparameter and epoch selection using validation loss, preserving the best model weights for final test evaluation.
- Save and reload model checkpoints.
- Introduce BatchNorm and Dropout, and study their effects on training and generalization.
- Compare architectures under matching experimental settings and investigate data augmentation to reduce overfitting.
