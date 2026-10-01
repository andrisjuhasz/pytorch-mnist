# PyTorch MNIST MLP

A simple multilayer perceptron for classifying handwritten digits from the MNIST dataset, implemented in PyTorch.

This is primarily an **educational project** developed while learning the core abstractions and training workflow of PyTorch. Rather than aiming for state-of-the-art MNIST performance, the project focuses on understanding how the different parts of a PyTorch training pipeline fit together.

The current implementation serves as a fully connected baseline. A planned next step is to implement a convolutional neural network (CNN) and compare it with this model.

## Model

The network is a small multilayer perceptron:

```text
28 × 28 image
    ↓
Flatten
    ↓
Linear(784 → 128)
    ↓
ReLU
    ↓
Linear(128 → 10)
```

Training uses cross-entropy loss and stochastic gradient descent (SGD).

## Training and evaluation

The MNIST training dataset is split into training and validation subsets using a fixed random seed for reproducibility.

The script:

1. trains the model using mini-batches;
2. evaluates loss and accuracy on the validation set after each epoch;
3. compares several learning-rate and momentum combinations;
4. records and plots the training and validation histories;
5. selects hyperparameters based on validation performance; and
6. evaluates the selected model on the held-out MNIST test set.

The experiments use the same model initialization and mini-batch ordering so that the different SGD hyperparameter settings can be compared under the same random conditions.

## Results

After 20 epochs of training, the three SGD configurations produced the following final-epoch metrics:

| Learning rate | Momentum | Training loss | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 0.40 | 0.0 | 0.0021 | 0.0735 | 97.88% |
| 0.15 | 0.5 | 0.0042 | 0.0777 | 97.88% |
| 0.03 | 0.9 | 0.0041 | 0.0791 | 97.88% |

Based on validation performance, the configuration with learning rate `0.4` and momentum `0.0` was selected for final evaluation.

Its performance on the held-out test set was:

- **Test loss:** 0.0730
- **Test accuracy:** 98.19%

## PyTorch concepts explored

The project was built as a way to learn PyTorch beyond simply using a pre-written training loop. In particular, it provided practical experience with:

- **Tensors and autograd** — tensor operations, computation graphs, gradient tracking, `backward()`, and parameter gradients.
- **`nn.Module`** — PyTorch's model abstraction and the use of modules such as `nn.Linear`, `nn.ReLU`, `nn.Flatten`, `nn.Sequential`, and `nn.CrossEntropyLoss`.
- **Datasets and `DataLoader`** — dataset access, train/validation splitting, batching, shuffling, and iteration over mini-batches.
- **Optimizers** — `torch.optim.Optimizer` and SGD, including the effects of learning rate and momentum.
- **Training and evaluation modes** — `model.train()`, `model.eval()`, and `torch.no_grad()`.
- **`torchvision`** — loading MNIST and applying input transformations with `transforms.Compose` and `transforms.ToTensor`.
- **Reproducibility** — explicit random generators and manual seeding for reproducible dataset splits and comparable experiments.
- **Validation and hyperparameter selection** — separating training, validation, and test data and using them for their distinct purposes.

## Requirements

- Python 3
- PyTorch
- torchvision
- Matplotlib

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

Run:

```bash
python mnist_mlp.py
```

MNIST is downloaded automatically on the first run.

The script prints the recorded training and validation metrics, displays plots comparing the hyperparameter configurations, and reports the final test loss and accuracy.

## Next steps

The next stage of the project will introduce a convolutional neural network and compare its performance with the fully connected baseline.
