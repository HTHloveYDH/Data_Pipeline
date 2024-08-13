import random

from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T
# import torchvision.transforms.v2 as T

from data_pipeline.transform.augmenters import augment_image


def get_custom_train_transform(input_size:tuple, version='V1'):
    if version == 'V1':
        composed_transforms = T.Compose(
            [
                # T.ToPILImage(), 
                T.Resize(input_size), T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5), T.RandomRotation(20),
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor 
                # of shape (C x H x W) in the range [0.0, 1.0] [if] the PIL Image belongs to one of the modes 
                # (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or [if] the numpy.ndarray has dtype = np.uint8
                # In the other cases, tensors are returned without scaling.
                # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
                T.ToTensor()
            ]
        )
        def transform(image:Image, **kwargs):
            return composed_transforms(image)
    elif version == 'V2':
        def transform(image:Image, **kwargs):
            image = image.resize(input_size, Image.BILINEAR)
            if random.uniform(0, 1) > 1.0 - 0.9:  # the probability of doing augmentation is 80%
                image = augment_image(image, **{'random_aug_config': kwargs['random_aug_config']})
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            image = image / kwargs['rescale_config']['scale'] + kwargs['rescale_config']['offset']
            return image
    elif version == 'V3':
        def transform(image:Image, **kwargs):
            if random.uniform(0, 1) > 1.0 - 0.9:  # the probability of doing augmentation is 80%
                image = augment_image(image, **{'random_aug_config': kwargs['random_aug_config']})
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            return image
    else:
        raise ValueError(f" {version} is not supported ")
    return transform

def get_custom_valid_transform(input_size:tuple, version='V1'):
    if version == 'V1':
        composed_transforms = T.Compose(
            [
                # T.ToPILImage(), 
                T.Resize(input_size), T.ToTensor()
            ]
        )
        def transform(image:Image, **kwargs):
            return composed_transforms(image)
    elif version == 'V2':
        def transform(image:Image, **kwargs):
            image = image.resize(input_size, Image.BILINEAR)
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            image = image / kwargs['rescale_config']['scale'] + kwargs['rescale_config']['offset']
            return image
    elif version == 'V3':
        def transform(image:Image, **kwargs):
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            return image
    else:
        raise ValueError(f" {version} is not supported")
    return transform

def get_custom_test_transform(input_size:tuple, version='V1'):
    if version == 'V1':
        composed_transforms = T.Compose(
            [
                # T.ToPILImage(), 
                T.Resize(input_size), T.ToTensor()
            ]
        )
        def transform(image:Image, **kwargs):
            return composed_transforms(image)
    elif version == 'V2':
        def transform(image:Image, **kwargs):
            image = image.resize(input_size, Image.BILINEAR)
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            image = image / kwargs['rescale_config']['scale'] + kwargs['rescale_config']['offset']
            return image
    elif version == 'V3':
        def transform(image:Image, **kwargs):
            image = T.functional.pil_to_tensor(image).to(dtype=torch.float32)
            return image
    else:
        raise ValueError(f" {version} is not supported")
    return transform