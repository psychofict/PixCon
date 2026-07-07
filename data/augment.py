"""Image-space transforms and CutMix for the PixCon pipeline."""

import math
import random

import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms as T


_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


# ----------------------------- PIL transforms -----------------------------

def crop(img, mask, size):
    w, h = img.size
    padw = max(size - w, 0); padh = max(size - h, 0)
    if padw or padh:
        img = T.functional.pad(img, (0, 0, padw, padh), fill=0)
        mask = T.functional.pad(mask, (0, 0, padw, padh), fill=255)
    w, h = img.size
    x = random.randint(0, w - size); y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size)), mask.crop((x, y, x + size, y + size))


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def resize(img, mask, base, ratio):
    w, h = img.size
    long_side = random.randint(int(base * ratio[0]), int(base * ratio[1]))
    if h > w:
        oh = long_side; ow = int(w * long_side / h + 0.5)
    else:
        ow = long_side; oh = int(h * long_side / w + 0.5)
    return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


def blur(img, p=0.5):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
    return img


def normalize(img, mask):
    img_t = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])(img)
    mask_t = torch.from_numpy(np.array(mask)).long()
    return img_t, mask_t


# ----------------------------- Tensor augment + CutMix -----------------------------

def strong_augment(images):
    """Strong color augmentation on a normalized [B,3,H,W] tensor."""
    device = images.device
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    imgs = (images * std + mean).clamp(0, 1)
    out = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        if random.random() < 0.8:
            img = T.functional.adjust_brightness(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_contrast(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_saturation(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_hue(img, random.uniform(-0.25, 0.25))
        if random.random() < 0.2:
            img = img.mean(dim=0, keepdim=True).expand_as(img)
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img = T.functional.gaussian_blur(img, k, random.uniform(0.1, 2.0))
        out.append(img)
    return ((torch.stack(out).clamp(0, 1) - mean) / std)


def _rand_bbox(h, w, lam):
    cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    cy, cx = random.randint(0, h), random.randint(0, w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def cutmix(images, pseudo, confidence):
    """Apply CutMix within a batch consistently across image + targets.

    A random rectangular region is pasted from a shuffled batch into every
    sample. Returns cloned tensors; inputs are not modified.
    """
    b, _, h, w = images.shape
    perm = torch.randperm(b, device=images.device)
    lam = random.random()
    y1, y2, x1, x2 = _rand_bbox(h, w, lam)
    images = images.clone(); pseudo = pseudo.clone(); confidence = confidence.clone()
    images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    pseudo[:, y1:y2, x1:x2] = pseudo[perm][:, y1:y2, x1:x2]
    confidence[:, y1:y2, x1:x2] = confidence[perm][:, y1:y2, x1:x2]
    return images, pseudo, confidence
