"""Patch-padded whole-image and sliding-window inference."""

import torch
import torch.nn.functional as F


def pad_to_multiple(img, multiple=14):
    """Reflect-pad a [.., H, W] tensor so H and W are divisible by `multiple`.

    Returns (padded_tensor, (orig_h, orig_w)).
    """
    h, w = img.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, (h, w)


@torch.no_grad()
def whole_inference(model, img, patch=14):
    """One forward; pad input to multiple of `patch`, crop back to original."""
    padded, (h, w) = pad_to_multiple(img, patch)
    logits = model(padded)
    return logits[..., :h, :w]


@torch.no_grad()
def slide_inference(model, img, crop_size, num_classes,
                    stride_ratio=2 / 3, patch=14):
    """Sliding-window over large images, accumulating softmax probabilities."""
    b, _, h, w = img.shape
    stride = max(int(crop_size * stride_ratio), 1)
    n_h = max((h - crop_size + stride - 1) // stride + 1, 1)
    n_w = max((w - crop_size + stride - 1) // stride + 1, 1)
    probs = img.new_zeros((b, num_classes, h, w))
    count = img.new_zeros((b, 1, h, w))
    for i in range(n_h):
        for j in range(n_w):
            y1 = min(i * stride, max(h - crop_size, 0))
            x1 = min(j * stride, max(w - crop_size, 0))
            y2, x2 = min(y1 + crop_size, h), min(x1 + crop_size, w)
            y1, x1 = max(y2 - crop_size, 0), max(x2 - crop_size, 0)
            window = img[:, :, y1:y2, x1:x2]
            padded, (wh, ww) = pad_to_multiple(window, patch)
            logits = model(padded)[..., :wh, :ww]
            probs[:, :, y1:y2, x1:x2] += F.softmax(logits, dim=1)
            count[:, :, y1:y2, x1:x2] += 1
    return probs / count.clamp(min=1)
