import cv2
import numpy as np


def grayscale_crop(obs):
    grayscale = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (80, 80), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, -1)
