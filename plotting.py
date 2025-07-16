import torchvision.transforms as transforms


def save_first_9_images_grid(trainset):
    from PIL import Image
    import os

    img_size = 28  # MNIST images are 28x28
    grid_size = 3
    bar_size = 9
    total_size = img_size * grid_size + bar_size * (grid_size - 1)
    grid_img = Image.new("L", (total_size, total_size), color=255)

    found = [False] * 9
    number = [0] * 9
    images = [None] * 9
    idx = 0
    while not all(found) and idx < len(trainset):
        img_tensor, label = trainset[idx]
        if 1 <= label <= 9 and not found[label - 1]:
            found[label - 1] = True
            images[label - 1] = img_tensor
            number[label - 1] = idx
        idx += 1

    for i in range(9):
        if images[i] is not None:
            img_tensor = images[i] * 1  # denormalize
            img = transforms.ToPILImage()(img_tensor)
            row = i // grid_size
            col = i % grid_size
            x = col * (img_size + bar_size)
            y = row * (img_size + bar_size)
            grid_img.paste(img, (x, y))

    # print(number)
    grid_img = grid_img.convert("RGB")
    os.makedirs("output", exist_ok=True)
    grid_img.save("output/mnist_3x3_grid.jpg", "JPEG")
    return number


def save_first_9_images_grid_pca(X_recon, filename="output/mnist_3x3_grid_pca.jpg"):
    from PIL import Image
    import os
    import numpy as np

    bar_size = 9
    img_size = 28
    grid_size = 3
    total_size = img_size * grid_size + bar_size * (grid_size - 1)
    grid_img = Image.new("L", (total_size, total_size), color=255)
    for idx in range(9):
        img = Image.fromarray((X_recon[idx] * 255).astype(np.uint8))
        row = idx // 3
        col = idx % 3
        x = col * (img_size + bar_size)
        y = row * (img_size + bar_size)
        grid_img.paste(img, (x, y))
    grid_img = grid_img.convert("RGB")
    os.makedirs("output", exist_ok=True)
    grid_img.save(filename, "JPEG")
    # print(f"Saved {filename}")
