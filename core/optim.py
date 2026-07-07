"""Optimizer construction.

UniMatch V2 uses a simple two-group AdamW: encoder LR vs decoder LR with no
layer-wise decay. We replicate that here. The PixCon projection head joins
the decoder group at the same LR.
"""

import torch


def build_optimizer(model, backbone_lr=5e-6, decoder_lr=2e-4,
                    weight_decay=0.01, betas=(0.9, 0.999)):
    """Two-group AdamW: backbone (low LR) + everything else (decoder LR).

    No weight decay on 1-D parameters, biases, or ViT tokens.
    """
    backbone_wd, backbone_nd, head_wd, head_nd = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = (
            p.ndim == 1
            or name.endswith('.bias')
            or 'cls_token' in name
            or 'pos_embed' in name
            or 'register_tokens' in name
            or 'mask_token' in name
        )
        is_backbone = name.startswith('backbone.')
        if is_backbone and no_decay:
            backbone_nd.append(p)
        elif is_backbone:
            backbone_wd.append(p)
        elif no_decay:
            head_nd.append(p)
        else:
            head_wd.append(p)

    groups = [
        {'params': backbone_wd, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': backbone_nd, 'lr': backbone_lr, 'weight_decay': 0.0},
        {'params': head_wd,     'lr': decoder_lr,  'weight_decay': weight_decay},
        {'params': head_nd,     'lr': decoder_lr,  'weight_decay': 0.0},
    ]
    return torch.optim.AdamW([g for g in groups if g['params']],
                             betas=betas, lr=decoder_lr)


def poly_lr_scale(current_iter, total_iters, power=0.9):
    return (1.0 - current_iter / max(total_iters, 1)) ** power
