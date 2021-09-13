import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


# taken from https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.shape)
    if min_size < size:
        ow, oh = img.shape
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    # def __call__(self, image, target):
    #     for t in self.transforms:
    #         image, target = t(image, target)
    #     return image, target

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        # target = F.resize(target, size, interpolation=Image.NEAREST)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            # target = F.hflip(target)
        return image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # image = pad_if_smaller(image, self.size)
        # target = pad_if_smaller(target, self.size, fill=255)
        channels = image.shape[2]
        cropped_image = np.ndarray(shape=(self.size, self.size, channels), dtype=image.dtype)
        crop_params = T.RandomCrop.get_params(Image.fromarray(image[:,:,0]), (self.size, self.size))
        for c in range(channels):
            image_channel = Image.fromarray(image[:, :, c])
            cropped_image_channel = F.crop(image_channel, *crop_params)
            cropped_image[:,:,c] = cropped_image_channel
        # target = F.crop(target, *crop_params)

        return cropped_image


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = Image.fromarray(image)
        image = F.center_crop(image, self.size)
        # target = F.center_crop(target, self.size)
        return image


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = F.resize(image, self.size)
        # target = F.resize(target, size, interpolation=Image.NEAREST)
        return image


class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
