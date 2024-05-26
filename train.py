import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from network import LeNet5, AlexNet


# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Use a fixed seed
SEED = 42
set_seed(SEED)

# Hyper-parameters
CLASSES = 10
EPOCH = 20
LR = 0.01

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Build dataset
def get_datasets(model_name):
    if model_name == "LeNet5":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    training_data = datasets.FashionMNIST(
        root="dataset",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = datasets.FashionMNIST(
        root="dataset",
        train=False,
        download=True,
        transform=test_transform,
    )

    return training_data, test_data

# Model list
best_acc_model = dict()
def train(model_class, model_name, batch_size):

    save_path = f"checkpoints/{model_name}_bs{batch_size}.pth"

    model = model_class(CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    training_data, test_data = get_datasets(model_name)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []
    best_acc = 0.0

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0
        num_correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            num_correct += (preds == labels).sum().item()
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_acc = num_correct / len(training_data)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        print(f"Epoch {epoch + 1}: Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")

        model.eval()
        test_loss = 0.0
        num_correct = 0

        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            num_correct += (preds == labels).sum().item()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

        test_acc = num_correct / len(test_data)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    best_acc_model[(model_name, batch_size)] = best_acc
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCH + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCH + 1), train_acces, label='Train Acc')
    plt.plot(range(1, EPOCH + 1), test_losses, label='Test Loss')
    plt.plot(range(1, EPOCH + 1), test_acces, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('{} (batch size={}): Training and Test Loss Over Epochs'.format(model_name, batch_size))
    plt.legend()
    if not os.path.exists('images'):
        os.makedirs('images')

    file_name = os.path.join('images', f'{model_name}_bs{batch_size}_loss_plot.png')
    plt.savefig(file_name)
    plt.close()
    plt.show()

if __name__ == "__main__":
    models = [LeNet5, AlexNet]
    batch_sizes = [4,8,16,32,64]

    for model_class in models:
        for batch_size in batch_sizes:
            model_name = model_class.__name__
            print(f"Training {model_name} with batch size {batch_size}")
            train(model_class, model_name, batch_size)

    print(best_acc_model)
