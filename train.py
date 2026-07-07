"""PixCon training: UniMatch V2 consistency + clean-positive contrastive auxiliary.

Single-purpose, single-scheme. No ResNet, no per-class threshold module, no
boundary loss. Just: DINOv2 backbone, DPT-lite decoder, two strong+CutMix views,
complementary channel dropout, fixed conf threshold 0.95, supervised InfoNCE
auxiliary against a clean-positive memory bank.
"""

import argparse
import json
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

# Reduce CUDA memory fragmentation before torch sees the env var.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from core.contrastive import (
    ClassMemoryBank, pixcon_loss, sample_anchors_from_labeled,
)
from core.ema import EMATeacher
from core.inference import slide_inference, whole_inference
from core.metrics import SegmentationMetrics
from core.optim import build_optimizer, poly_lr_scale
from data.augment import cutmix, strong_augment
from data.dataset import SemiDataset, build_loaders
from model.segmentor import PixConSegmentor


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--labeled-id-path', required=True)
    p.add_argument('--unlabeled-id-path', required=True)
    p.add_argument('--val-id-path', default=None,
                   help='If omitted, uses dataset/splits/<name>/val.txt under the repo root.')
    p.add_argument('--save-path', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--no-pixcon', action='store_true',
                   help='UniMatch V2 baseline (no contrastive auxiliary).')
    p.add_argument('--pixcon-bank-filter', default='clean',
                   choices=['clean', 'labeled', 'conf'],
                   help="Bank admission rule (ablation / decomposition): "
                        "'clean' (PixCon: labeled & pred==GT), "
                        "'labeled' (labeled only, rho_F=0 without correctness), "
                        "'conf' (ReCo/U2PL-style confidence bank, rho_F>0).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if args.val_id_path is None:
        # Convention: repo root holds dataset/splits/<name>/val.txt
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)
        args.val_id_path = os.path.join(repo_root, 'dataset', 'splits',
                                        args.dataset, 'val.txt')
    return args


def validate(model, valloader, num_classes, dataset, crop_size, device):
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    use_slide = (dataset == 'cityscapes')
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                             enabled=(device.type == 'cuda')):
        for batch in valloader:
            imgs, masks, *_ = batch
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            if use_slide:
                out = slide_inference(model, imgs, crop_size, num_classes)
            else:
                out = whole_inference(model, imgs)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            metrics.add_batch(pred, masks_np)
    return metrics.evaluate()


def train_one_epoch(student, teacher, labeled_loader, unlabeled_loader,
                    optimizer, ce, bank, args, device, epoch, tb, global_step):
    student.train(); teacher.model.eval()
    conf = float(args.conf_thresh)
    use_pixcon = (bank is not None) and (not args.no_pixcon)

    total = {'loss': 0.0, 'loss_x': 0.0, 'loss_u': 0.0,
             'loss_pix': 0.0, 'mask': 0.0}
    total_iters = len(unlabeled_loader) * args.epochs
    base = epoch * len(unlabeled_loader)
    log_every = max(1, len(unlabeled_loader) // 8)
    lit = iter(labeled_loader)

    amp = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

    for i, (img_u, _) in enumerate(unlabeled_loader):
        try:
            img_l, mask_l = next(lit)
        except StopIteration:
            lit = iter(labeled_loader)
            img_l, mask_l = next(lit)
        img_l = img_l.to(device); mask_l = mask_l.to(device); img_u = img_u.to(device)

        # Supervised + optional PixCon embeddings on the labeled batch.
        with amp:
            if use_pixcon:
                logits_l, z_l = student.forward_features(img_l)
            else:
                logits_l = student(img_l)
                z_l = None
            loss_x = ce(logits_l, mask_l)

        loss_pix = logits_l.new_zeros(())
        if use_pixcon:
            with amp:
                B, D, h2, w2 = z_l.shape
                gt_lo = F.interpolate(mask_l.unsqueeze(1).float(), size=(h2, w2),
                                      mode='nearest').squeeze(1).long()
                with torch.no_grad():
                    logits_lo = F.interpolate(logits_l.float(), size=(h2, w2),
                                              mode='bilinear', align_corners=True)
                    pred_lo = logits_lo.argmax(1)
                    conf_lo = (torch.softmax(logits_lo, dim=1).max(1).values
                               if args.pixcon_bank_filter == 'conf' else None)
                a_f, a_l = sample_anchors_from_labeled(
                    z_l, gt_lo, pred_lo,
                    max_per_class=int(args.pixcon_per_class),
                    bank_filter=str(args.pixcon_bank_filter),
                    conf=conf_lo, conf_thresh=float(args.conf_thresh))
            if a_f.shape[0] > 0:
                loss_pix = pixcon_loss(
                    a_f, a_l, bank,
                    temperature=float(args.pixcon_temp),
                    max_anchors=int(args.pixcon_max_anchors))
                bank.enqueue(a_f, a_l)
        del logits_l, z_l, img_l, mask_l

        # Weak-view teacher pseudo-labels.
        with torch.no_grad(), amp:
            t_logits = teacher(img_u)
            pseudo_w = torch.argmax(t_logits, dim=1)
            conf_w = torch.max(F.softmax(t_logits, dim=1), dim=1).values
            del t_logits

        # Two independent strong+CutMix views.
        s1 = strong_augment(img_u); s2 = strong_augment(img_u)
        del img_u
        s1, p1, c1 = cutmix(s1, pseudo_w, conf_w)
        s2, p2, c2 = cutmix(s2, pseudo_w, conf_w)
        del pseudo_w, conf_w

        with amp:
            x_cat = torch.cat([s1, s2], dim=0)
            del s1, s2
            pred_s1, pred_s2 = student.forward_dual(x_cat)
            del x_cat

            ce1 = F.cross_entropy(pred_s1, p1, reduction='none', ignore_index=255)
            ce2 = F.cross_entropy(pred_s2, p2, reduction='none', ignore_index=255)
            keep1 = (c1 >= conf).float(); keep2 = (c2 >= conf).float()
            denom1 = max(int((p1 != 255).sum().item()), 1)
            denom2 = max(int((p2 != 255).sum().item()), 1)
            l1 = (ce1 * keep1).sum() / denom1
            l2 = (ce2 * keep2).sum() / denom2
            loss_u = (l1 + l2) / 2.0
            loss = (loss_x + loss_u) / 2.0
            if use_pixcon:
                loss = loss + float(args.pixcon_weight) * loss_pix
            mask_ratio = 0.5 * (keep1.mean().item() + keep2.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total['loss'] += loss.item(); total['loss_x'] += loss_x.item()
        total['loss_u'] += loss_u.item()
        total['loss_pix'] += float(loss_pix.item()) if use_pixcon else 0.0
        total['mask'] += mask_ratio

        # Poly LR + EMA ramp-up.
        cur = base + i
        scale = poly_lr_scale(cur, total_iters)
        for g in optimizer.param_groups:
            if 'base_lr' not in g:
                g['base_lr'] = g['lr']
            g['lr'] = g['base_lr'] * scale
        teacher.decay = min(1.0 - 1.0 / (cur + 1), float(args.ema_decay_max))
        teacher.update(student)

        if tb:
            step = global_step + i
            tb.add_scalar('train/loss', loss.item(), step)
            tb.add_scalar('train/loss_x', loss_x.item(), step)
            tb.add_scalar('train/loss_u', loss_u.item(), step)
            tb.add_scalar('train/mask', mask_ratio, step)
            if use_pixcon:
                tb.add_scalar('train/loss_pix', float(loss_pix.item()), step)
        if i % log_every == 0:
            pix_str = f' Lpix:{float(loss_pix.item()):.3f}' if use_pixcon else ''
            logger.info(
                f'iter {i}/{len(unlabeled_loader)} LR:{optimizer.param_groups[-1]["lr"]:.2e} '
                f'L:{loss.item():.3f} Lx:{loss_x.item():.3f} Lu:{loss_u.item():.3f}{pix_str} '
                f'mask:{mask_ratio:.3f}'
            )

    n = len(unlabeled_loader)
    return {k: v / n for k, v in total.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.save_path, 'out.log')),
                                  logging.StreamHandler()], force=True)
    logger.info(f'Args: {vars(args)}')
    tb = SummaryWriter(log_dir=args.save_path)

    student = PixConSegmentor(
        backbone=args.backbone, nclass=args.nclass,
        decoder_dim=int(args.decoder_dim), proj_dim=int(args.proj_dim),
    ).to(device)
    teacher = EMATeacher(student, decay=float(args.ema_decay_max)).to(device)
    optimizer = build_optimizer(student,
                                backbone_lr=float(args.backbone_lr),
                                decoder_lr=float(args.decoder_lr),
                                weight_decay=float(args.weight_decay))
    ce = CrossEntropyLoss(ignore_index=255).to(device)
    bank = (None if args.no_pixcon
            else ClassMemoryBank(args.nclass, dim=int(args.proj_dim),
                                 size_per_class=int(args.pixcon_bank_size),
                                 device=device))

    labeled_loader, unlabeled_loader, valloader = build_loaders(
        args.dataset, args.data_root,
        args.labeled_id_path, args.unlabeled_id_path, args.val_id_path,
        crop_size=int(args.crop_size), batch_size=int(args.batch_size),
    )
    logger.info(f'Labeled: {len(labeled_loader.dataset)}  '
                f'Unlabeled: {len(unlabeled_loader.dataset)}  '
                f'Val: {len(valloader.dataset)}  '
                f'Iters/epoch: {len(unlabeled_loader)}')

    start_epoch = 0
    best_miou_ema = 0.0; best_epoch_ema = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        student.load_state_dict(ckpt['student'], strict=True)
        teacher.load_state_dict(ckpt['teacher'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except ValueError:
            logger.warning('optimizer state mismatch; skipping')
        start_epoch = ckpt.get('epoch', 0) + 1
        best_miou_ema = ckpt.get('best_miou_ema', 0.0)
        best_epoch_ema = ckpt.get('best_epoch_ema', 0)
        if bank is not None and 'bank' in ckpt:
            bank.queue = ckpt['bank']['queue'].to(device)
            bank.ptr = ckpt['bank']['ptr'].to(device)
            bank.filled = ckpt['bank']['filled'].to(device)
        logger.info(f'Resumed from epoch {start_epoch}, best EMA {best_miou_ema:.2f}')

    for epoch in range(start_epoch, int(args.epochs)):
        logger.info(f'\nEpoch {epoch + 1}/{args.epochs}')
        t0 = time.time()
        gs = epoch * len(unlabeled_loader)
        stats = train_one_epoch(student, teacher, labeled_loader, unlabeled_loader,
                                optimizer, ce, bank, args, device, epoch, tb, gs)
        per_class, miou = validate(student, valloader, int(args.nclass),
                                   args.dataset, int(args.crop_size), device)
        per_class_ema, miou_ema = validate(teacher.model, valloader, int(args.nclass),
                                           args.dataset, int(args.crop_size), device)
        miou_pct = float(miou) * 100; miou_ema_pct = float(miou_ema) * 100
        logger.info(f'  Student {miou_pct:.2f}  EMA {miou_ema_pct:.2f}  '
                    f'L {stats["loss"]:.3f}  mask {stats["mask"]:.3f}  '
                    f'{(time.time() - t0) / 60:.1f} min')

        new_best = miou_ema_pct > best_miou_ema
        if new_best:
            best_miou_ema = miou_ema_pct; best_epoch_ema = epoch + 1

        ckpt = {
            'epoch': epoch,
            'best_miou_ema': best_miou_ema,
            'best_epoch_ema': best_epoch_ema,
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if bank is not None:
            ckpt['bank'] = {'queue': bank.queue, 'ptr': bank.ptr, 'filled': bank.filled}
        torch.save(ckpt, os.path.join(args.save_path, 'latest.pth'))
        if new_best:
            torch.save(ckpt, os.path.join(args.save_path, 'best_ema.pth'))
            logger.info(f'  >>> New best EMA: {best_miou_ema:.2f} (ep {best_epoch_ema})')

        tb.add_scalar('eval/miou', miou_pct, epoch + 1)
        tb.add_scalar('eval/miou_ema', miou_ema_pct, epoch + 1)
        tb.flush()

    logger.info(f'\nDone. Best EMA: {best_miou_ema:.2f} (ep {best_epoch_ema})')
    with open(os.path.join(args.save_path, 'results.json'), 'w') as f:
        json.dump({'best_miou_ema': best_miou_ema,
                   'best_epoch_ema': best_epoch_ema,
                   'args': vars(args)}, f, indent=2)


if __name__ == '__main__':
    main()
