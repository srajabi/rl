import numpy as np


def grayscale_crop(obs):

    downsampled = obs[::2, ::2]
    mean = np.mean(downsampled, axis=2).astype(np.uint8)

    return mean[20:-5, :]


def grayscale(obs):
    downsampled = obs[::2, ::2]
    return np.mean(downsampled, axis=2).astype(np.uint8)
