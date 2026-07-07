"""DINOv2 + DPT-lite + integrated PixCon projection head.

One model exposes:
  - forward(x): standard segmentation logits.
  - forward_dual(x):    UniMatch V2 two-stream forward with complementary
                        channel dropout. x is [2B, 3, H, W]; returns
                        (logits_s1, logits_s2).
  - forward_features(x): segmentation logits + projected pixel embeddings.
                         Used by PixCon to compute the contrastive auxiliary
                         on labeled-batch features.

No legacy from earlier per-class threshold / boundary-loss code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2Backbone


class _ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(self.act(x)); y = self.bn1(y)
        y = self.conv2(self.act(y)); y = self.bn2(y)
        return y + x


class _FusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.rcu_skip = _ResidualConvUnit(c)
        self.rcu_out = _ResidualConvUnit(c)

    def forward(self, x, skip=None):
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = x + self.rcu_skip(skip)
        x = self.rcu_out(x)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class PixConSegmentor(nn.Module):
    """DINOv2 backbone + DPT-lite decoder + segmentation head + projection head.

    Args:
        backbone: 'dinov2_vits14' | 'dinov2_vitb14' | 'dinov2_vitl14'
            or a DINOv2Backbone instance.
        nclass: number of segmentation classes.
        decoder_dim: common decoder channel width (default 256).
        proj_dim: PixCon projection head output dim (default 256).
        pretrained: load pretrained DINOv2 weights from torch.hub.
    """

    def __init__(self, backbone='dinov2_vitb14', nclass=21,
                 decoder_dim=256, proj_dim=256, pretrained=True):
        super().__init__()
        self.backbone = (DINOv2Backbone(backbone, pretrained=pretrained)
                         if isinstance(backbone, str) else backbone)
        ch = self.backbone.channels
        assert len(ch) == 4

        # Project each backbone layer to the decoder width.
        self.projects = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1, bias=False)
                                       for c in ch])
        # Resample 4 same-resolution ViT maps into a coarse->fine pyramid.
        self.resamples = nn.ModuleList([
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 4, stride=4),   # 4x up
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2),   # 2x up
            nn.Identity(),                                                # keep
            nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1),  # 2x down
        ])
        self.fusions = nn.ModuleList([_FusionBlock(decoder_dim) for _ in range(4)])

        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, nclass, 1),
        )

        # PixCon projection head: 1x1 conv -> BN -> ReLU -> 1x1 conv -> L2 norm.
        self.proj_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, proj_dim, 1),
        )

    def _decode(self, x):
        """Returns the fused decoder feature [B, C, H', W']."""
        feats = self.backbone(x)
        pyr = [self.resamples[i](self.projects[i](feats[i])) for i in range(4)]
        y = self.fusions[3](pyr[3])
        y = self.fusions[2](y, pyr[2])
        y = self.fusions[1](y, pyr[1])
        y = self.fusions[0](y, pyr[0])
        return y

    def forward(self, x):
        """Plain segmentation forward."""
        h, w = x.shape[-2:]
        fused = self._decode(x)
        logits = self.head(fused)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)

    def forward_features(self, x):
        """Segmentation logits + L2-normalised pixel embeddings (for PixCon)."""
        h, w = x.shape[-2:]
        fused = self._decode(x)
        logits = self.head(fused)
        z = F.normalize(self.proj_head(fused), dim=1)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        return logits, z

    def forward_dual(self, x, p_drop=0.5):
        """UniMatch V2 dual stream with complementary channel dropout.

        x: [2B, 3, H, W] (two strong views concatenated along batch dim).
        Returns (logits_s1, logits_s2), each [B, nclass, H, W].
        """
        if x.shape[0] % 2 != 0:
            raise ValueError('forward_dual expects 2B batch (two views concatenated)')
        h, w = x.shape[-2:]
        b = x.shape[0] // 2
        fused = self._decode(x)
        c = fused.shape[1]
        mask = (torch.rand(1, c, 1, 1, device=fused.device) < p_drop).float()
        s1 = self.head(fused[:b] * (mask * 2.0))
        s2 = self.head(fused[b:] * ((1.0 - mask) * 2.0))
        s1 = F.interpolate(s1, size=(h, w), mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=(h, w), mode='bilinear', align_corners=True)
        return s1, s2
