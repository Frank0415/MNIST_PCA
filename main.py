import torch
import torchvision
import torchvision.transforms as transforms
import plotting
import pca
import train
import torch.nn as nn
import torch.optim as optim
import time
from cnn_train import SimpleCNN, train_model, evaluate_model
from nn_model import MNISTNN

CNN_FLAG = False

def run_experiment(max_iterations, pca_components=196):
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
            if iteration % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}], Iteration [{iteration}], Loss: {loss.item():.4f}"
                )
        if iteration >= max_iterations:
            break
    nn_time = time.time() - start
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
    print(f"NN original: accuracy={acc_nn:.4f}, time={nn_time:.2f}s")

    # --- CNN on original data ---
    
    if CNN_FLAG:
        print("--- Training CNN on original data ---")
        cnn = SimpleCNN()
        optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
        cnn_time = train_model(
            cnn, trainloader, criterion, optimizer_cnn, max_iter=max_iterations
        )
        acc_cnn = evaluate_model(cnn, testloader)
        print(f"CNN original: accuracy={acc_cnn:.4f}, time={cnn_time:.2f}s")

    # --- PCA ---
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
            if iteration % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}], Iteration [{iteration}], Loss: {loss.item():.4f}"
                )
        if iteration >= max_iterations:
            break
    nn_pca_time = time.time() - start
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
    print(f"NN PCA: accuracy={acc_nn_pca:.4f}, time={nn_pca_time:.2f}s")

    # --- CNN on PCA data (reshape to 14x14 if 196 components) ---
    
    if CNN_FLAG:
    
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
        print(f"CNN PCA: accuracy={acc_cnn_pca:.4f}, time={cnn_pca_time:.2f}s")

        var_ratio = sum(pca_model.explained_variance_ratio_[:pca_components])
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
        
        pca_components_picture = 512 # DO NOT CHANGE
        
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


def main():
    for max_iter in [1000, 3000]:
        run_experiment(max_iter)


if __name__ == "__main__":
    main()
