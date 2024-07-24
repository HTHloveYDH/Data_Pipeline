import torchvision.transforms as T
from torchvision.transforms.v2 import CutMix, MixUp
from PIL import Image


def get_affine_augmenter(probs:list):
    random_rotation = T.RandomRotation(degrees=(0, 25))
    random_zoom = T.RandomAffine(degrees=(0, 0), translate=None, scale=(0.75, 1.25), shear=None)
    random_translation = T.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15), scale=None, shear=None)
    random_vertical_flip = T.RandomHorizontalFlip(p=1.0)
    random_horizontal_flip = T.RandomVerticalFlip(p=1.0)
    random_shear = T.RandomAffine(degrees=(0, 0), translate=None, scale=None, shear=(0.5, 1.5))
    random_3d_rotation = T.RandomPerspective(distortion_scale=0.6, p=1.0)
    # random_3d_rotation = 
    rot90 = lambda x: T.functional.rotate(x, angle=90)
    rot180 = lambda x: T.RandomRotation(x, angle=180)
    rot270 = lambda x: T.RandomRotation(x, angle=270)
    affine_operations = [
        random_rotation, random_zoom, random_translation, random_vertical_flip, random_horizontal_flip, 
        random_shear, random_3d_rotation, rot90, rot180, rot270
    ]
    affine_augmenter = T.RandomChoice(affine_operations, p=probs)
    return affine_augmenter


def get_quality_augmenter(probs:list):
    random_brightness = T.ColorJitter(brightness=.5, saturation=.0, hue=.0)
    random_contrast =  T.RandomAutocontrast()
    random_hue = T.ColorJitter(brightness=.0, saturation=.0, hue=.3)
    random_saturation = T.ColorJitter(brightness=.0, saturation=.5, hue=.0)
    rgb_to_hsv = lambda x: x
    random_jpeg_quality = lambda x: x
    inverse_color = T.RandomInvert()
    gaussian_noise = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)) 
    random_channel_shift = lambda x: x
    random_sharpness = T.RandomAdjustSharpness(sharpness_factor=2)
    posterization = T.RandomPosterize(bits=4)
    grid_mask = GridMask(10, 30)
    random_color_degeneration = lambda x: x
    quality_operations = [
        random_brightness, random_contrast, random_hue, random_saturation, rgb_to_hsv, random_jpeg_quality, 
        inverse_color, gaussian_noise, random_channel_shift, random_sharpness, posterization, grid_mask, 
        random_color_degeneration
    ]
    quality_augmenter = T.RandomChoice(quality_operations, p=probs)
    return quality_augmenter


def get_mix_augmenter(label_type:str, num_classes_map:dict, probs:dict):
    mixup = MixUp()
    cutmix = CutMix()
    mix_augmenter = T.RandomChoice([mixup, cutmix, lambda x, y: (x, y)], p=probs)
    return mix_augmenter


def get_zoom_augmenter(probs:dict):
    zoom_in = T.Compose(
        [
            T.RandomAffine(degrees=(0, 0), translate=None, scale=(1.0, 1.25), shear=None), 
            T.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15), scale=None, shear=None)
        ]
    )
    zoom_out = T.Compose(
        [
            T.RandomAffine(degrees=(0, 0), translate=None, scale=(0.75, 1.0), shear=None), 
            T.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15), scale=None, shear=None)
        ]
    )
    zoom_operations = [zoom_in, zoom_out]
    zoom_augmenter = T.RandomChoice(zoom_operations, p=probs['mode'])
    return zoom_augmenter


def get_hybrid_augmenter(probs:dict):
    affine_augmenter = get_affine_augmenter(probs['affine_augmenter'])
    quality_augmenter = get_quality_augmenter(probs['quality_augmenter'])
    zoom_augmenter = get_zoom_augmenter(probs['zoom_augmenter'])
    def dual(image:Image, probs=probs['hybrid_augmenter']['dual']):
        dual_hybrid_0 = T.Compose([affine_augmenter, quality_augmenter])
        dual_hybrid_1 = T.Compose([affine_augmenter, zoom_augmenter])
        dual_hybrid_2 = T.Compose([quality_augmenter, affine_augmenter])
        dual_hybrid_3 = T.Compose([quality_augmenter, zoom_augmenter])
        dual_hybrid_4 = T.Compose([zoom_augmenter, affine_augmenter])
        dual_hybrid_5 = T.Compose([zoom_augmenter, quality_augmenter])
        dual_hybrid_6 = T.Compose([quality_augmenter, quality_augmenter])
        dual_operations = [
            dual_hybrid_0, dual_hybrid_1, dual_hybrid_2, dual_hybrid_3, dual_hybrid_4, dual_hybrid_5, 
            dual_hybrid_6
        ]
        image = T.RandomChoice([dual_operations])(image)
        return image
    def triple(image:Image, probs=probs['hybrid_augmenter']['triple']):
        triple_hybrid_0 = T.Compose([affine_augmenter, quality_augmenter, zoom_augmenter])
        triple_hybrid_1 = T.Compose([affine_augmenter, zoom_augmenter, quality_augmenter])
        triple_hybrid_2 = T.Compose([quality_augmenter, affine_augmenter, zoom_augmenter])
        triple_hybrid_3 = T.Compose([quality_augmenter, zoom_augmenter, affine_augmenter])
        triple_hybrid_4 = T.Compose([zoom_augmenter, affine_augmenter, quality_augmenter])
        triple_hybrid_5 = T.Compose([zoom_augmenter, quality_augmenter, affine_augmenter])
        triple_hybrid_6 = T.Compose([affine_augmenter, quality_augmenter, quality_augmenter])
        triple_hybrid_7 = T.Compose([quality_augmenter, affine_augmenter, quality_augmenter])
        triple_hybrid_8 = T.Compose([quality_augmenter, quality_augmenter, affine_augmenter])
        triple_hybrid_9 = T.Compose([zoom_augmenter, quality_augmenter, quality_augmenter])
        triple_hybrid_10 = T.Compose([quality_augmenter, zoom_augmenter, quality_augmenter])
        triple_hybrid_11 = T.Compose([quality_augmenter, quality_augmenter, zoom_augmenter])
        triple_hybrid_12 = T.Compose([quality_augmenter, quality_augmenter, quality_augmenter])
        triple_operations = [
            triple_hybrid_0, triple_hybrid_1, triple_hybrid_2, triple_hybrid_3, triple_hybrid_4, triple_hybrid_5, 
            triple_hybrid_6, triple_hybrid_7, triple_hybrid_8, triple_hybrid_9, triple_hybrid_10, triple_hybrid_11,
            triple_hybrid_12
        ]
        image = T.RandomChoice([triple_operations], p=probs)(image)
        return image
    hybrid_operations = [dual, triple]
    hybrid_augmenter = T.RandomChoice(hybrid_operations, p=probs['hybrid_augmenter']['mode'])
    return hybrid_augmenter


def augment_image(image:Image, random_aug_config:dict):
    affine_augmenter = get_affine_augmenter(random_aug_config['affine_aug_probs'])
    quality_augmenter = get_quality_augmenter(random_aug_config['quality_aug_probs'])
    zoom_augmenter = get_zoom_augmenter(random_aug_config['zoom_aug_probs'])
    hybrid_augmenter = get_hybrid_augmenter(random_aug_config['hybrid_aug_probs'])
    augmenters = [affine_augmenter, quality_augmenter, zoom_augmenter, hybrid_augmenter]
    augmenter = T.RandomChoice(augmenters, p=random_aug_config[''])
    image = augmenter(image)
    return image