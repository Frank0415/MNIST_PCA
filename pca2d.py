import numpy as np


def TwoDPCA(imgs, dim):
    a, b, c = imgs.shape
    average = np.zeros((b, c))
    for i in range(a):
        average += imgs[i, :, :] / (a * 1.0)
    G_t = np.zeros((c, c))
    for j in range(a):
        img = imgs[j, :, :]
        temp = img - average
        G_t = G_t + np.dot(temp.T, temp) / (a * 1.0)
    w, v = np.linalg.eigh(G_t)
    w = w[::-1]
    v = v[:, ::-1]
    print("alpha={}".format(sum(w[:dim]) * 1.0 / sum(w)))
    u = v[:, :dim]
    print("u_shape:{}".format(u.shape))
    return u  # u is the projection matrix


def TTwoDPCA(imgs, dim):
    u = TwoDPCA(imgs, dim)
    a1, b1, c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i, :, :], u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img, dim)
    print("uu_shape:{}".format(uu.shape))
    return u, uu


def image_2D2DPCA(images, u, uu):
    a, b, c = images.shape
    new_images = np.ones((a, uu.shape[1], u.shape[1]))
    for i in range(a):
        Y = np.dot(uu.T, images[i, :, :])
        Y = np.dot(Y, u)
        new_images[i, :, :] = Y
    return new_images
