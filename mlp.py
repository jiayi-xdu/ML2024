import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyper-parameters
BATCH_SIZE = 4
CLASSES = 10
EPOCH = 10
LR = 0.01

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Build dataset
training_data = datasets.FashionMNIST(
    root="dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root="dataset",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Build dataloader
train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# print(training_data[0][0].shape)

# Construct model architecture
class MLP(nn.Module):
    def __init__(self, num_class):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

model = MLP(CLASSES).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    model.train()
    train_acc, num_correct = 0., 0
    train_loss, test_loss = 0., 0.

    # Training loop with progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.reshape(imgs.shape[0], -1)
            outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            num_correct += (preds == labels).sum().item()
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            tepoch.set_postfix(loss=train_loss / len(train_loader), accuracy=num_correct / len(training_data))

    train_acc = num_correct / len(training_data)
    train_loss = train_loss / len(training_data)
    print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")

    model.eval()
    num_correct = 0
    test_loss = 0.

    # Testing loop with progress bar
    with tqdm(test_loader, unit="batch") as tepoch:
        for imgs, labels in tepoch:
            tepoch.set_description(f"Testing")
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.reshape(imgs.shape[0], -1)
            with torch.no_grad():
                outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            num_correct += (preds == labels).sum().item()
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            tepoch.set_postfix(loss=test_loss / len(test_loader), accuracy=num_correct / len(test_data))

    test_acc = num_correct / len(test_data)
    test_loss = test_loss / len(test_data)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Show result
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

cols, rows = 3, 3
figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
figure.tight_layout()

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[sample_idx]
    img = img.unsqueeze(0).to(device)  # Add a batch dimension and move to device
    model.eval()
    with torch.no_grad():
        output = model(img.view(1, -1))
    prediction = output.argmax(dim=1, keepdim=True).item()

    # 获取标签名称
    true_label_name = labels_map[label]
    pred_label_name = labels_map[prediction]

    # 绘制图像和标题
    ax = figure.add_subplot(rows, cols, i)
    ax.set_title(f"True: {true_label_name}\nPred: {pred_label_name}", fontsize=10)
    ax.axis("off")
    ax.imshow(img.cpu().squeeze(), cmap="gray")

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()