"""CPU-only smoke tests for the clean PixCon codebase.

Uses a fake DINOv2 backbone (with .vit + .blocks, no torch.hub download) so
the segmentor, contrastive head, memory bank, loss, and end-to-end training
step can all be exercised without GPU or internet. Run from the pixcon/
directory:

    python -m pytest tests/ -q
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from core.contrastive import (
    ClassMemoryBank, pixcon_loss, sample_anchors_from_labeled,
)
from core.ema import EMATeacher
from core.inference import slide_inference, whole_inference
from core.optim import build_optimizer
from data.augment import cutmix, strong_augment
from model.segmentor import PixConSegmentor


EMBED, DEPTH = 768, 12
PATCH = 14
NCLASS = 21


class _FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED, PATCH, stride=PATCH)
        self.blocks = nn.ModuleList([nn.Linear(EMBED, EMBED) for _ in range(DEPTH)])


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = EMBED
        self.patch_size = PATCH
        self.channels = [EMBED] * 4
        self.vit = _FakeViT()

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % PATCH or w % PATCH:
            raise ValueError(f'{h}x{w} not divisible by {PATCH}')
        tok = self.vit.patch_embed(x)
        return [tok + i for i in range(4)]


def _model():
    return PixConSegmentor(backbone=_FakeBackbone(), nclass=NCLASS,
                           decoder_dim=128, proj_dim=64)


def test_forward_shapes():
    m = _model(); m.eval()
    x = torch.randn(2, 3, 84, 84)
    with torch.no_grad():
        logits = m(x)
    assert logits.shape == (2, NCLASS, 84, 84)
    logits, z = m.forward_features(x)
    assert logits.shape == (2, NCLASS, 84, 84)
    # z is at decoder resolution; just check norms.
    n = z.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(n, torch.ones_like(n), atol=1e-4)


def test_forward_dual_diverges():
    m = _model(); m.eval()
    x = torch.randn(4, 3, 84, 84)  # 2 views * 2 = 4
    with torch.no_grad():
        s1, s2 = m.forward_dual(x)
    assert s1.shape == (2, NCLASS, 84, 84)
    assert s2.shape == (2, NCLASS, 84, 84)
    assert not torch.allclose(s1, s2), 'complementary dropout should diverge streams'


def test_bank_loss_grad_flow():
    bank = ClassMemoryBank(NCLASS, dim=16, size_per_class=32, device='cpu')
    for k in range(NCLASS):
        f = torch.randn(8, 16); f = f / f.pow(2).sum(dim=1, keepdim=True).sqrt()
        bank.enqueue(f, torch.full((8,), k, dtype=torch.long))
    a = torch.randn(40, 16, requires_grad=True)
    an = a / a.pow(2).sum(dim=1, keepdim=True).sqrt()
    l = pixcon_loss(an, torch.randint(0, NCLASS, (40,)), bank, temperature=0.1)
    assert torch.isfinite(l) and l.item() > 0
    l.backward()
    assert torch.isfinite(a.grad).all()


def test_train_step_endtoend():
    """One iteration of the full training step on a tiny fake batch."""
    device = torch.device('cpu')
    m = _model(); teacher = EMATeacher(m, decay=0.99)
    opt = build_optimizer(m, backbone_lr=1e-4, decoder_lr=1e-3)
    ce = CrossEntropyLoss(ignore_index=255)
    bank = ClassMemoryBank(NCLASS, dim=64, size_per_class=16, device='cpu')

    img_l = torch.randn(2, 3, 70, 70)
    mask_l = torch.randint(0, NCLASS, (2, 70, 70))
    img_u = torch.randn(2, 3, 70, 70)

    # Supervised + PixCon embeddings
    logits_l, z_l = m.forward_features(img_l)
    loss_x = ce(logits_l, mask_l)
    import torch.nn.functional as F
    gt_lo = F.interpolate(mask_l.unsqueeze(1).float(), size=z_l.shape[-2:],
                          mode='nearest').squeeze(1).long()
    pred_lo = F.interpolate(logits_l, size=z_l.shape[-2:],
                            mode='bilinear', align_corners=True).argmax(1)
    a_f, a_l = sample_anchors_from_labeled(z_l, gt_lo, pred_lo, max_per_class=8)
    loss_pix = pixcon_loss(a_f, a_l, bank) if a_f.shape[0] > 0 else loss_x * 0
    bank.enqueue(a_f, a_l)

    # Teacher pseudo-labels + two strong+CutMix views
    with torch.no_grad():
        t_logits = teacher(img_u)
    pseudo = torch.argmax(t_logits, dim=1)
    conf = torch.max(F.softmax(t_logits, dim=1), dim=1).values
    s1 = strong_augment(img_u); s2 = strong_augment(img_u)
    s1, p1, c1 = cutmix(s1, pseudo, conf)
    s2, p2, c2 = cutmix(s2, pseudo, conf)
    pred_s1, pred_s2 = m.forward_dual(torch.cat([s1, s2]))
    ce1 = F.cross_entropy(pred_s1, p1, reduction='none', ignore_index=255)
    ce2 = F.cross_entropy(pred_s2, p2, reduction='none', ignore_index=255)
    keep1 = (c1 >= 0.5).float(); keep2 = (c2 >= 0.5).float()
    denom1 = max(int((p1 != 255).sum().item()), 1)
    denom2 = max(int((p2 != 255).sum().item()), 1)
    loss_u = ((ce1 * keep1).sum() / denom1 + (ce2 * keep2).sum() / denom2) / 2
    loss = (loss_x + loss_u) / 2 + 0.1 * loss_pix

    opt.zero_grad(); loss.backward(); opt.step()
    teacher.update(m)
    assert torch.isfinite(loss)


def test_inference_helpers():
    m = _model(); m.eval()
    img = torch.randn(1, 3, 100, 130)  # not divisible by 14
    out = whole_inference(m, img)
    assert out.shape == (1, NCLASS, 100, 130)
    out2 = slide_inference(m, torch.randn(1, 3, 180, 200),
                           crop_size=70, num_classes=NCLASS)
    assert out2.shape == (1, NCLASS, 180, 200)
    # softmax accumulator sums to ~1 per pixel.
    s = out2.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)


if __name__ == '__main__':
    test_forward_shapes()
    test_forward_dual_diverges()
    test_bank_loss_grad_flow()
    test_train_step_endtoend()
    test_inference_helpers()
    print('All PixCon clean-folder smoke tests passed.')
