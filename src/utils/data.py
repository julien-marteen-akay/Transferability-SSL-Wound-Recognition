from PIL import Image
import torch
from torchvision import transforms
from typing import Union


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


def get_augmentation_transforms():
    transform = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.95, 1.05))
    ]
    return transform


def pad_to_square(img):
    width, height = img.size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2

    # Convert PIL image to PyTorch tensor
    img_tensor = transforms.functional.to_tensor(img)

    # Pad the image using PyTorch operations
    padded_img = torch.zeros((3, max_dim, max_dim))
    padded_img[:, pad_height:pad_height + height, pad_width:pad_width + width] = img_tensor

    # Convert PyTorch tensor to PIL image
    padded_img = transforms.functional.to_pil_image(padded_img)

    return padded_img


def get_transforms(augmentations=None, keep_aspect_ratio=False):
    # mean = [0.5, 0.5, 0.5]  # Mean value for each channel
    # std = [0.5, 0.5, 0.5]   # Standard deviation for each channel

    transform = [
        transforms.Lambda(lambda path: load_image(path)),
        transforms.Resize((224, 224))
    ]
    if keep_aspect_ratio:
        transform.insert(1, transforms.Lambda(lambda img: pad_to_square(img)))
    if augmentations:
        transform.extend(augmentations)

    transform = transforms.Compose([
        *transform,
        transforms.ToTensor(),  # this seems to normalize to [0, 1] already. Pretty useful, no need to normalize
        # transforms.Normalize(mean, std)
    ])
    return transform
