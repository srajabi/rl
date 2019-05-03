import gym
import matplotlib.pyplot as plt
import numpy as np
import timeit
import functools
from unittest import TestCase
from agents.img_processing.cv2_obs_processing import grayscale_crop
from agents.img_processing.skimage_obs_processing import grayscale_crop as skimage_grayscale_crop
from agents.img_processing.np_obs_processing import grayscale_crop as np_grayscale_crop, grayscale

class ObsPreprocessingTest(TestCase):

    def test_image_processing_cv2(self):
        env = gym.make('BreakoutDeterministic-v4')
        env.reset()

        for _ in range(100):
            env.step(env.action_space.sample())

        image, _, _, _ = env.step(1)

        processed_image = grayscale_crop(image)
        processed_image2 = skimage_grayscale_crop(image)
        processed_image3 = np_grayscale_crop(image)
        processed_image4 = grayscale(image)

        print('orig',image.shape)
        print('cv2',processed_image.shape)
        print('skimage',processed_image2.shape)
        print('grayscale_crop',processed_image3.shape)
        print('grayscale',processed_image4.shape)

        print('cv2',timeit.timeit(functools.partial(grayscale_crop, image), number=1000))
        print('skimage',timeit.timeit(functools.partial(skimage_grayscale_crop, image), number=1000))
        print('grayscale_crop',timeit.timeit(functools.partial(np_grayscale_crop, image), number=1000))
        print('grayscale',timeit.timeit(functools.partial(grayscale, image), number=1000))

        f, axes = plt.subplots(3, 2)

        axes[0, 0].imshow(image)
        axes[0, 1].imshow(np.squeeze(processed_image), cmap='gray', vmin=0, vmax=255)

        axes[1, 0].imshow(np.squeeze(processed_image2), cmap='gray', vmin=0, vmax=255)
        axes[1, 1].imshow(np.squeeze(processed_image3), cmap='gray', vmin=0, vmax=255)

        axes[2, 0].imshow(np.squeeze(processed_image4), cmap='gray', vmin=0, vmax=255)
        axes[2, 1].imshow(image)
        plt.show()
