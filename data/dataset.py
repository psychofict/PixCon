"""Semi-supervised segmentation dataset.

Three modes:
  - 'labeled':   read labeled split file, return (img, mask) with full augmentation
  - 'unlabeled': read unlabeled split file, return (img, dummy_mask)
  - 'val':       read val split file, return (img, mask, id) with only normalisation
"""

import math
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .augment import (
    crop, hflip, resize, normalize,
)


PASCAL_BASE = 400
CITYSCAPES_BASE = 2048
ADE20K_BASE = 520


class SemiDataset(Dataset):
    """Minimal dataset for labeled / unlabeled / val modes.

    Each line of a split file is "RELATIVE_IMG_PATH RELATIVE_MASK_PATH" (mask path
    optional for unlabeled mode). The transforms are applied identically in
    'labeled' and 'unlabeled' modes (random resize + crop + hflip + normalise);
    strong augmentation and CutMix happen on the GPU inside the training loop.
    """

    def __init__(self, name, root, mode, crop_size, id_path):
        self.name = name
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        with open(id_path) as f:
            self.ids = [l.strip() for l in f if l.strip()]
        self.base_size = {
            'pascal': PASCAL_BASE,
            'cityscapes': CITYSCAPES_BASE,
            'ade20k': ADE20K_BASE,
        }.get(name, PASCAL_BASE)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        line = self.ids[idx]
        parts = line.split(' ')
        img = Image.open(os.path.join(self.root, parts[0])).convert('RGB')

        if self.mode == 'val':
            mask = Image.open(os.path.join(self.root, parts[1]))
            img_t, mask_t = normalize(img, mask)
            return img_t, mask_t, line

        if self.mode == 'unlabeled':
            mask = Image.fromarray(np.full((img.size[1], img.size[0]), 255, dtype=np.uint8))
        else:
            mask = Image.open(os.path.join(self.root, parts[1]))

        img, mask = resize(img, mask, self.base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.crop_size)
        img, mask = hflip(img, mask, p=0.5)
        return normalize(img, mask)


def build_loaders(name, root, labeled_path, unlabeled_path, val_path,
                  crop_size, batch_size, num_workers=4):
    """Three DataLoaders: labeled (oversampled), unlabeled, val."""
    from torch.utils.data import DataLoader

    lset = SemiDataset(name, root, 'labeled', crop_size, labeled_path)
    uset = SemiDataset(name, root, 'unlabeled', crop_size, unlabeled_path)
    # Oversample labeled to match unlabeled length (UM2 pattern).
    if len(lset.ids) < len(uset.ids):
        repeat = math.ceil(len(uset.ids) / len(lset.ids))
        lset.ids = (lset.ids * repeat)[:len(uset.ids)]

    vset = SemiDataset(name, root, 'val', crop_size, val_path)

    labeled = DataLoader(lset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=True, drop_last=True)
    unlabeled = DataLoader(uset, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True)
    val_batch = 4 if name == 'cityscapes' else 1
    val = DataLoader(vset, batch_size=val_batch, shuffle=False,
                     num_workers=1, pin_memory=True, drop_last=False)
    return labeled, unlabeled, val
