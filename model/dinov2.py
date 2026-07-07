"""DINOv2 ViT backbone — extracts 4 intermediate feature maps.

Loads via torch.hub from facebookresearch/dinov2. Patch size 14, so inputs must
be divisible by 14. All transformer blocks operate at the same spatial
resolution H/14 x W/14; the decoder builds the feature pyramid.
"""

import torch
import torch.nn as nn

_VARIANTS = {
    'dinov2_vits14': {'embed_dim': 384,  'depth': 12, 'layers': (2, 5, 8, 11)},
    'dinov2_vitb14': {'embed_dim': 768,  'depth': 12, 'layers': (2, 5, 8, 11)},
    'dinov2_vitl14': {'embed_dim': 1024, 'depth': 24, 'layers': (4, 11, 17, 23)},
    'dinov2_vitg14': {'embed_dim': 1536, 'depth': 40, 'layers': (9, 19, 29, 39)},
}
PATCH_SIZE = 14


class DINOv2Backbone(nn.Module):
    """Returns 4 patch-token feature maps reshaped to (B, C, H/14, W/14)."""

    def __init__(self, name='dinov2_vitb14', pretrained=True, out_layers=None):
        super().__init__()
        if name not in _VARIANTS:
            raise ValueError(f'unknown variant {name!r}; choose from {list(_VARIANTS)}')
        spec = _VARIANTS[name]
        self.name = name
        self.embed_dim = spec['embed_dim']
        self.out_layers = tuple(out_layers) if out_layers is not None else spec['layers']
        self.patch_size = PATCH_SIZE
        self.channels = [self.embed_dim] * len(self.out_layers)
        self.vit = torch.hub.load('facebookresearch/dinov2', name, pretrained=pretrained)

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f'input {h}x{w} not divisible by patch size {self.patch_size}')
        return list(self.vit.get_intermediate_layers(
            x, n=self.out_layers, reshape=True, return_class_token=False, norm=True))
