import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


def grayscale_crop(obs):
    return np.uint8(
        resize(rgb2gray(obs), (80, 80), mode='constant') * 255
    )
