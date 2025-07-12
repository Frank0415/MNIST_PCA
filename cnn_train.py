import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from train import SoftmaxRegression, prepare_data
import pca


class SimpleCNN(nn.Module):
    def __init__(self, input_size=28):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear((input_size // 4) * (input_size // 4) * 64, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)


def train_model(model, trainloader, criterion, optimizer, max_iter=1000):
    model.train()
    start = time.time()
    iteration = 0
    for epoch in range(100):
        for inputs, labels in trainloader:
            if iteration >= max_iter:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iteration += 1
        if iteration >= max_iter:
            break
    end = time.time()
    return end - start


def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    # --- CNN on original data ---
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    cnn = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    print("Training CNN on original data...")
    cnn_time = train_model(cnn, trainloader, criterion, optimizer, max_iter=1000)
    cnn_acc = evaluate_model(cnn, testloader)
    print(f"CNN original: accuracy={cnn_acc:.4f}, time={cnn_time:.2f}s")
    # --- Softmax on original data ---
    X_train, y_train = prepare_data(trainset)
    X_test, y_test = prepare_data(testset)
    softmax = SoftmaxRegression()
    optimizer_s = optim.SGD(softmax.parameters(), lr=0.01)
    print("Training Softmax on original data...")
    start = time.time()
    for i in range(1000):
        idx = np.random.choice(len(X_train), 100, replace=False)
        batch_xs = torch.tensor(X_train[idx], dtype=torch.float32)
        batch_ys = torch.tensor(y_train[idx], dtype=torch.long)
        optimizer_s.zero_grad()
        outputs = softmax(batch_xs)
        loss = criterion(outputs, batch_ys)
        loss.backward()
        optimizer_s.step()
    end = time.time()
    with torch.no_grad():
        outputs = softmax(torch.tensor(X_test, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        acc_softmax = (predicted == torch.tensor(y_test)).float().mean().item()
    print(f"Softmax original: accuracy={acc_softmax:.4f}, time={end - start:.2f}s")
    # --- PCA ---
    print("Fitting PCA...")
    pca_model = pca.fit_pca(trainset, n_components=100, model_path=None)
    X_train_pca = pca_model.transform(X_train)[:, :100]
    X_test_pca = pca_model.transform(X_test)[:, :100]

    # --- CNN on PCA data (reshape to 10x10) ---
    def make_pca_loader(X, y):
        X_img = torch.tensor(X, dtype=torch.float32).view(-1, 1, 10, 10)
        y = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_img, y)
        return torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

    pca_trainloader = make_pca_loader(X_train_pca, y_train)
    pca_testloader = make_pca_loader(X_test_pca, y_test)
    cnn_pca = SimpleCNN(input_size=10)
    optimizer_cnn_pca = optim.Adam(cnn_pca.parameters(), lr=0.0001)
    print("Training CNN on PCA data...")
    cnn_pca_time = train_model(
        cnn_pca, pca_trainloader, criterion, optimizer_cnn_pca, max_iter=1000
    )
    cnn_pca_acc = evaluate_model(cnn_pca, pca_testloader)
    print(f"CNN PCA: accuracy={cnn_pca_acc:.4f}, time={cnn_pca_time:.2f}s")
    # --- Softmax on PCA data ---
    softmax_pca = SoftmaxRegression(input_dim=100)
    optimizer_s_pca = optim.SGD(softmax_pca.parameters(), lr=0.01)
    print("Training Softmax on PCA data...")
    start = time.time()
    for i in range(1000):
        idx = np.random.choice(len(X_train_pca), 100, replace=False)
        batch_xs = torch.tensor(X_train_pca[idx], dtype=torch.float32)
        batch_ys = torch.tensor(y_train[idx], dtype=torch.long)
        optimizer_s_pca.zero_grad()
        outputs = softmax_pca(batch_xs)
        loss = criterion(outputs, batch_ys)
        loss.backward()
        optimizer_s_pca.step()
    end = time.time()
    with torch.no_grad():
        outputs = softmax_pca(torch.tensor(X_test_pca, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        acc_softmax_pca = (predicted == torch.tensor(y_test)).float().mean().item()
    print(f"Softmax PCA: accuracy={acc_softmax_pca:.4f}, time={end - start:.2f}s")


if __name__ == "__main__":
    main()
