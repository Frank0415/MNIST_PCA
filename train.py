import numpy as np
import torch
import os


class SoftmaxRegression(torch.nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)


def prepare_data(dataset):
    # Explicitly flatten images and collect labels, as in PCA_with_mnist
    X = []
    y = []
    for img, label in dataset:
        X.append(np.array(img).reshape(-1))
        y.append(int(label))
    X = np.stack(X)
    y = np.array(y)
    return X, y


def batch_iter(X, y, batch_size=100, shuffle=True):
    # Numpy-based batching, similar to PCA_with_mnist
    idxs = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_idx = idxs[start:end]
        yield X[batch_idx], y[batch_idx]


def train_classifier(
    model,
    X_train,
    y_train,
    criterion,
    optimizer,
    max_iterations=1000,
    batch_size=100,
    model_path=None,
):
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training softmax regression model...")
        model.train()
        iteration = 0
        while iteration < max_iterations:
            for X_batch, y_batch in batch_iter(
                X_train, y_train, batch_size=batch_size, shuffle=True
            ):
                if iteration >= max_iterations:
                    break
                inputs = torch.tensor(X_batch, dtype=torch.float32)
                labels = torch.tensor(y_batch, dtype=torch.long)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                iteration += 1
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
        print("Finished Training")
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved model to {model_path}")
    return model


def evaluate_classifier(model, X_test, y_test, batch_size=100):
    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in batch_iter(
            X_test, y_test, batch_size=batch_size, shuffle=False
        ):
            inputs = torch.tensor(X_batch, dtype=torch.float32)
            labels = torch.tensor(y_batch, dtype=torch.long)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy:.2f} %")
    return accuracy / 100.0


# For PCA, just use the same functions but with PCA-transformed data
