import torch
import torchvision
import torchvision.transforms as transforms
import plotting
import pca
import train
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from cnn_train import SimpleCNN, train_model, evaluate_model
from nn_model import MNISTNN
import pca2d
import sys

CNN_FLAG = True
VERBOSE = False


def run_experiment(
    max_iterations, pca_components=196, use_2dpca=False, row_top=14, col_top=14
):
    if VERBOSE:
        print(f"\n=== Training for {max_iterations} iterations ===")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0,), (1,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    image_indices = plotting.save_first_9_images_grid(trainset)

    # --- NN on original data ---
    if VERBOSE:
        print("--- Training NN on original data ---")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    model = MNISTNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    start = time.time()
    iteration = 0
    for epoch in range(100):
        for i, data in enumerate(trainloader, 0):
            if iteration >= max_iterations:
                break
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iteration += 1
            if VERBOSE and iteration % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}], Iteration [{iteration}], Loss: {loss.item():.4f}"
                )
        if iteration >= max_iterations:
            break
    nn_time = time.time() - start
    print(
        f"NN original done: time={nn_time:.2f}s, iter={max_iterations}",
        file=sys.stderr,
    )
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc_nn = correct / total
    if VERBOSE:
        print(f"NN original: accuracy={acc_nn:.4f}, time={nn_time:.2f}s")

    # --- CNN on original data ---

    cnn_time = None
    acc_cnn = None
    if CNN_FLAG:
        if VERBOSE:
            print("--- Training CNN on original data ---")
        cnn = SimpleCNN()
        optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
        cnn_time = train_model(
            cnn, trainloader, criterion, optimizer_cnn, max_iter=max_iterations
        )
        acc_cnn = evaluate_model(cnn, testloader)
        print(
            f"CNN original done: time={cnn_time:.2f}s, iter={max_iterations}",
            file=sys.stderr,
        )
        if VERBOSE:
            print(f"CNN original: accuracy={acc_cnn:.4f}, time={cnn_time:.2f}s")

    # --- PCA ---
    if VERBOSE:
        print("\n--- Training on PCA-transformed data ---")
    X_train_np, y_train_np = train.prepare_data(trainset)
    X_test_np, y_test_np = train.prepare_data(testset)
    pca_model = pca.fit_pca(trainset, n_components=784)
    X_train_pca = pca_model.transform(X_train_np)[:, :pca_components]
    X_test_pca = pca_model.transform(X_test_np)[:, :pca_components]

    # DataLoader for PCA data
    def make_pca_loader(X, y, size):
        X_img = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_img, y)
        return torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    pca_trainloader = make_pca_loader(X_train_pca, y_train_np, pca_components)
    pca_testloader = make_pca_loader(X_test_pca, y_test_np, pca_components)
    model_pca = nn.Sequential(
        nn.Linear(pca_components, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    optimizer_pca = optim.SGD(model_pca.parameters(), lr=0.1)
    start = time.time()
    iteration = 0
    for epoch in range(100):
        for i, data in enumerate(pca_trainloader, 0):
            if iteration >= max_iterations:
                break
            inputs, labels = data
            optimizer_pca.zero_grad()
            outputs = model_pca(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_pca.step()
            iteration += 1
            if VERBOSE and iteration % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}], Iteration [{iteration}], Loss: {loss.item():.4f}"
                )
        if iteration >= max_iterations:
            break
    nn_pca_time = time.time() - start
    print(
        f"NN PCA done: time={nn_pca_time:.2f}s, iter={max_iterations}",
        file=sys.stderr,
    )
    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in pca_testloader:
            images, labels = data
            outputs = model_pca(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc_nn_pca = correct / total
    if VERBOSE:
        print(f"NN PCA: accuracy={acc_nn_pca:.4f}, time={nn_pca_time:.2f}s")

    # --- CNN on PCA data (reshape to 14x14 if 196 components) ---

    cnn_pca_time = None
    acc_cnn_pca = None
    if CNN_FLAG:
        if VERBOSE:
            print("--- Training CNN on PCA data ---")

        def make_pca_loader_cnn(X, y, size):
            X_img = torch.tensor(X, dtype=torch.float32).view(-1, 1, size, size)
            y = torch.tensor(y, dtype=torch.long)
            dataset = torch.utils.data.TensorDataset(X_img, y)
            return torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

        side = int(pca_components**0.5)
        pca_trainloader_cnn = make_pca_loader_cnn(X_train_pca, y_train_np, side)
        pca_testloader_cnn = make_pca_loader_cnn(X_test_pca, y_test_np, side)
        cnn_pca = SimpleCNN(input_size=side)
        optimizer_cnn_pca = optim.Adam(cnn_pca.parameters(), lr=0.001)
        cnn_pca_time = train_model(
            cnn_pca,
            pca_trainloader_cnn,
            criterion,
            optimizer_cnn_pca,
            max_iter=max_iterations,
        )
        acc_cnn_pca = evaluate_model(cnn_pca, pca_testloader_cnn)
        print(
            f"CNN PCA done: time={cnn_pca_time:.2f}s, iter={max_iterations}",
            file=sys.stderr,
        )
        if VERBOSE:
            print(f"CNN PCA: accuracy={acc_cnn_pca:.4f}, time={cnn_pca_time:.2f}s")

        var_ratio = sum(pca_model.explained_variance_ratio_[:pca_components])
        if VERBOSE:
            print(f"Explained variance by {pca_components} PCs: {var_ratio:.4f}")

    if max_iterations == 1000:
        # # Reconstruct and save images 1-9 using all PCA components
        # X_recon_full = pca.reconstruct_images_pca(
        #     pca_model, trainset, n_components_to_use=784, image_indices=image_indices
        # )
        # plotting.save_first_9_images_grid_pca(
        #     X_recon_full,
        #     filename=f"output/mnist_3x3_grid_pca_full_1-9_{max_iterations}.jpg",
        # )

        pca_components_picture = 512  # DO NOT CHANGE

        # Reconstruct and save images 1-9 using selected PCA components
        X_recon_pca = pca.reconstruct_images_pca(
            pca_model,
            trainset,
            n_components_to_use=pca_components_picture,
            image_indices=image_indices,
        )
        plotting.save_first_9_images_grid_pca(
            X_recon_pca,
            filename=f"output/mnist_3x3_grid_pca_{pca_components_picture}_1-9_{max_iterations}.jpg",
        )

    # --- 2D PCA ---
    acc_nn_2dpca = None
    nn_2dpca_time = None
    acc_cnn_2dpca = None
    cnn_2dpca_time = None
    if use_2dpca:
        if VERBOSE:
            print("\n--- Training on 2D PCA-transformed data ---")
        # Prepare data as [N, 28, 28]
        X_train_np, y_train_np = train.prepare_data(trainset)
        X_test_np, y_test_np = train.prepare_data(testset)
        X_train_imgs = X_train_np.reshape(-1, 28, 28)
        X_test_imgs = X_test_np.reshape(-1, 28, 28)
        # Fit 2D PCA
        u, uu = pca2d.TTwoDPCA(X_train_imgs, row_top)
        X_train_2dpca = pca2d.image_2D2DPCA(X_train_imgs, u, uu)
        X_test_2dpca = pca2d.image_2D2DPCA(X_test_imgs, u, uu)
        # For NN: flatten
        X_train_2dpca_flat = X_train_2dpca.reshape(X_train_2dpca.shape[0], -1)
        X_test_2dpca_flat = X_test_2dpca.reshape(X_test_2dpca.shape[0], -1)
        # For CNN: reshape to [N, 1, row_top, col_top]
        X_train_2dpca_cnn = X_train_2dpca[:, np.newaxis, :, :]
        X_test_2dpca_cnn = X_test_2dpca[:, np.newaxis, :, :]
        # NN on 2D PCA
        model_2dpca = nn.Sequential(
            nn.Linear(row_top * col_top, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        optimizer_2dpca = optim.SGD(model_2dpca.parameters(), lr=0.1)
        criterion_2dpca = nn.CrossEntropyLoss()
        start = time.time()
        iteration = 0
        for epoch in range(100):
            for i in range(0, len(X_train_2dpca_flat), 100):
                if iteration >= max_iterations:
                    break
                batch_x = torch.tensor(
                    X_train_2dpca_flat[i : i + 100], dtype=torch.float32
                )
                batch_y = torch.tensor(y_train_np[i : i + 100], dtype=torch.long)
                optimizer_2dpca.zero_grad()
                outputs = model_2dpca(batch_x)
                loss = criterion_2dpca(outputs, batch_y)
                loss.backward()
                optimizer_2dpca.step()
                iteration += 1
                if VERBOSE and iteration % 500 == 0:
                    print(f"2D PCA NN Iteration {iteration}, Loss: {loss.item():.4f}")
            if iteration >= max_iterations:
                break
        nn_2dpca_time = time.time() - start
        print(
            f"NN 2D PCA done: time={nn_2dpca_time:.2f}s, iter={max_iterations}",
            file=sys.stderr,
        )
        # Evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X_test_2dpca_flat), 100):
                batch_x = torch.tensor(
                    X_test_2dpca_flat[i : i + 100], dtype=torch.float32
                )
                batch_y = torch.tensor(y_test_np[i : i + 100], dtype=torch.long)
                outputs = model_2dpca(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        acc_nn_2dpca = correct / total
        if VERBOSE:
            print(f"NN 2D PCA: accuracy={acc_nn_2dpca:.4f}, time={nn_2dpca_time:.2f}s")
        if CNN_FLAG:
            # CNN on 2D PCA
            cnn_2dpca = SimpleCNN(input_size=row_top)
            optimizer_cnn_2dpca = optim.Adam(cnn_2dpca.parameters(), lr=0.001)
            criterion_cnn_2dpca = nn.CrossEntropyLoss()
            from torch.utils.data import TensorDataset, DataLoader

            trainset_2dpca = TensorDataset(
                torch.tensor(X_train_2dpca_cnn, dtype=torch.float32),
                torch.tensor(y_train_np, dtype=torch.long),
            )
            testset_2dpca = TensorDataset(
                torch.tensor(X_test_2dpca_cnn, dtype=torch.float32),
                torch.tensor(y_test_np, dtype=torch.long),
            )
            trainloader_2dpca = DataLoader(trainset_2dpca, batch_size=50, shuffle=True)
            testloader_2dpca = DataLoader(testset_2dpca, batch_size=100, shuffle=False)
            start = time.time()
            cnn_2dpca_time = train_model(
                cnn_2dpca,
                trainloader_2dpca,
                criterion_cnn_2dpca,
                optimizer_cnn_2dpca,
                max_iter=max_iterations,
            )
            acc_cnn_2dpca = evaluate_model(cnn_2dpca, testloader_2dpca)
            print(
                f"CNN 2D PCA done: time={cnn_2dpca_time:.2f}s, iter={max_iterations}",
                file=sys.stderr,
            )
            if VERBOSE:
                print(
                    f"CNN 2D PCA: accuracy={acc_cnn_2dpca:.4f}, time={cnn_2dpca_time:.2f}s"
                )
    return {
        "iters": max_iterations,
        "nn_acc": acc_nn,
        "nn_time": nn_time,
        "cnn_acc": acc_cnn,
        "cnn_time": cnn_time,
        "nn_pca_acc": acc_nn_pca,
        "nn_pca_time": nn_pca_time,
        "cnn_pca_acc": acc_cnn_pca,
        "cnn_pca_time": cnn_pca_time,
        "nn_2dpca_acc": acc_nn_2dpca,
        "nn_2dpca_time": nn_2dpca_time,
        "cnn_2dpca_acc": acc_cnn_2dpca,
        "cnn_2dpca_time": cnn_2dpca_time,
    }


def main():
    results = []
    for max_iter in [1000, 3000]:
        res = run_experiment(max_iter, use_2dpca=True)
        results.append(res)
    print("\nSummary Table:")
    if CNN_FLAG:
        print(
            "Iters | NN Orig Acc | NN Orig Time | CNN Orig Acc | CNN Orig Time | NN PCA Acc | NN PCA Time | CNN PCA Acc | CNN PCA Time | NN 2D PCA Acc | NN 2D PCA Time | CNN 2D PCA Acc | CNN 2D PCA Time"
        )
        for row in results:
            print(
                f"{row['iters']:>5} | {row['nn_acc']:.4f} | {row['nn_time']:.2f} | "
                f"{row['cnn_acc'] if row['cnn_acc'] is not None else 'N/A':>10} | {row['cnn_time'] if row['cnn_time'] is not None else 'N/A':>10} | "
                f"{row['nn_pca_acc']:.4f} | {row['nn_pca_time']:.2f} | "
                f"{row['cnn_pca_acc'] if row['cnn_pca_acc'] is not None else 'N/A':>10} | {row['cnn_pca_time'] if row['cnn_pca_time'] is not None else 'N/A':>10} | "
                f"{row['nn_2dpca_acc'] if row['nn_2dpca_acc'] is not None else 'N/A':>10} | {row['nn_2dpca_time'] if row['nn_2dpca_time'] is not None else 'N/A':>10} | "
                f"{row['cnn_2dpca_acc'] if row['cnn_2dpca_acc'] is not None else 'N/A':>10} | {row['cnn_2dpca_time'] if row['cnn_2dpca_time'] is not None else 'N/A':>10}"
            )
    else:
        print(
            "Iters | NN Orig Acc | NN Orig Time | NN PCA Acc | NN PCA Time | NN 2D PCA Acc | NN 2D PCA Time"
        )
        for row in results:
            print(
                f"{row['iters']:>5} | {row['nn_acc']:.4f} | {row['nn_time']:.2f} | "
                f"{row['nn_pca_acc']:.4f} | {row['nn_pca_time']:.2f} | "
                f"{row['nn_2dpca_acc'] if row['nn_2dpca_acc'] is not None else 'N/A':>10} | {row['nn_2dpca_time'] if row['nn_2dpca_time'] is not None else 'N/A':>10}"
            )


if __name__ == "__main__":
    main()
