import numpy as np
from sklearn.decomposition import PCA


def fit_pca(trainset, n_components=None):
    print("Fitting PCA model...")
    # Flatten images and stack into a matrix
    X = np.stack([np.array(img[0]).reshape(-1) for img in trainset])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def transform_pca(pca, dataset):
    X = np.stack([np.array(img[0]).reshape(-1) for img in dataset])
    return pca.transform(X)


def inverse_transform_pca(pca, X_pca):
    return pca.inverse_transform(X_pca)


def explained_variance_pca(pca):
    return np.sum(pca.explained_variance_ratio_)


def reconstruct_images_pca(
    pca, dataset, n_components_to_use, n_images=9, image_indices=None
):
    # Collect the images as numpy arrays
    X = []
    if image_indices:
        for i in image_indices:
            img_tensor, _ = dataset[i]
            X.append(np.array(img_tensor).reshape(-1))
    else:
        count = 0
        idx = 0
        while count < n_images and idx < len(dataset):
            img_tensor, _ = dataset[idx]
            X.append(np.array(img_tensor).reshape(-1))
            count += 1
            idx += 1

    if not X:
        return np.array([])

    X = np.stack(X)

    X_pca = pca.transform(X)

    X_pca_zeroed = np.zeros_like(X_pca)
    X_pca_zeroed[:, :n_components_to_use] = X_pca[:, :n_components_to_use]
    X_recon = pca.inverse_transform(X_pca_zeroed)

    return X_recon.reshape((len(X), 28, 28))
