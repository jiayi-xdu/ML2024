import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from network import LeNet5, LeNet6, LeNet7, LeNet8, LeNet9, AlexNet, AlexNet_dp2, AlexNet_dp7, VGG, vgg, LeNet5_bn, \
    LeNet5_dp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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
BATCH_SIZE = 16
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
training_data = datasets.FashionMNIST(
    root="dataset",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="dataset",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

# Build dataloader
train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Model list

best_acc_model = dict()


def train(model_class, model_name, channel_config=None):
    save_path = f"checkpoints/{model_name}.pth"

    # ch1, ch2 = channel_config
    model = model_class(CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []
    best_acc = 0.0

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0
        num_correct = 0

        # Training loop with progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            for imgs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                imgs, labels = imgs.to(device), labels.to(device)
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
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acces.append(train_acc)
        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")

        model.eval()
        test_loss = 0.0
        num_correct = 0

        # Testing loop with progress bar
        with tqdm(test_loader, unit="batch") as tepoch:
            for imgs, labels in tepoch:
                tepoch.set_description(f"Testing")
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                _, preds = torch.max(outputs, -1)
                num_correct += (preds == labels).sum().item()
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                tepoch.set_postfix(loss=test_loss / len(test_loader), accuracy=num_correct / len(test_data))

        test_acc = num_correct / len(test_data)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        test_acces.append(test_acc)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    # best_acc_model[model_name] = best_acc
    best_acc_model[channel_config] = best_acc
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCH + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCH + 1), train_acces, label='Train Acc')
    plt.plot(range(1, EPOCH + 1), test_losses, label='Test Loss')
    plt.plot(range(1, EPOCH + 1), test_acces, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('{}: Training and Test Loss Over Epochs'.format(model_name))
    plt.legend()
    # 检查 images 文件夹是否存在，不存在则创建
    if not os.path.exists('images'):
        os.makedirs('images')

    # 保存图片
    file_name = os.path.join('images', f'{model_name}_loss_plot.png')
    plt.savefig(file_name)
    plt.close()
    plt.show()

    # Show result
    # labels_map = {
    #     0: "T-Shirt",
    #     1: "Trouser",
    #     2: "Pullover",
    #     3: "Dress",
    #     4: "Coat",
    #     5: "Sandal",
    #     6: "Shirt",
    #     7: "Sneaker",
    #     8: "Bag",
    #     9: "Ankle Boot",
    # }
    #
    # cols, rows = 3, 3
    # figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8, 8))
    #
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(test_data), size=(1,)).item()
    #     img, label = test_data[sample_idx]
    #     img = img.unsqueeze(0).to(device)  # Add a batch dimension and move to device
    #     model.eval()
    #     with torch.no_grad():
    #         output = model(img)
    #     prediction = output.argmax(dim=1, keepdim=True).item()
    #
    #     # 获取标签名称
    #     true_label_name = labels_map[label]
    #     pred_label_name = labels_map[prediction]
    #
    #     # 绘制图像和标题
    #     ax = figure.add_subplot(rows, cols, i)
    #     ax.set_title(f"True: {true_label_name}\nPred: {pred_label_name}", fontsize=10, pad=10)
    #     ax.axis("off")
    #     ax.imshow(img.cpu().squeeze(), cmap="gray")
    #
    # figure.suptitle("{} prediction".format(model_name), fontsize=16, fontweight='bold')
    # # 调整子图之间的间距
    # plt.subplots_adjust(hspace=0.7, wspace=0.5, top=0.9, bottom=0.1)
    # plt.show()

    # 检查 images 文件夹是否存在，不存在则创建
    # if not os.path.exists('images'):
    #     os.makedirs('images')
    #
    # # 保存图片
    # file_name = os.path.join('images', f'{model_name}_predictions.png')
    # figure.savefig(file_name)


if __name__ == "__main__":
    models = [LeNet5, LeNet6, LeNet7, LeNet8, LeNet9]

    # ch_config = [(48,64),
    #              (48,128),
    #              (96,256)
    #              ]

    for model_class in models:
        # TODO: 对输入网络类型做判断
        model_name = model_class.__name__
        print(f"Training {model_name}")
        train(model_class, model_name)
    print(best_acc_model)
