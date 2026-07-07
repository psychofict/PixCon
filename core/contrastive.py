"""PixCon: clean-positive memory bank + supervised contrastive loss.

The bank is updated only from LABELED pixels where the student prediction
matches the ground-truth label -- a clean-positive filter strictly stronger
than confidence. The InfoNCE loss is then evaluated against this clean bank
with class-balanced anchor sampling.

Reference relatives: ReCo (Liu et al., 2022), U2PL (Wang et al., 2022),
SupCon (Khosla et al., 2020). Distinguishing factor: bank construction.
"""

import torch
import torch.nn.functional as F


class ClassMemoryBank:
    """Per-class FIFO queue of unit-norm pixel embeddings, on-device."""

    def __init__(self, num_classes, dim, size_per_class=256, device='cuda'):
        self.num_classes = num_classes
        self.dim = dim
        self.size = size_per_class
        self.device = device
        self.queue = torch.zeros(num_classes, size_per_class, dim, device=device)
        self.ptr = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.filled = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def enqueue(self, features, labels):
        """Push (features, labels) into per-class queues.

        Args:
            features: [N, D] L2-normalised features (detached automatically).
            labels: [N] class indices; out-of-range entries are skipped.
        """
        features = features.detach()
        for k in range(self.num_classes):
            mask = labels == k
            if not mask.any():
                continue
            fk = features[mask]
            nk = fk.shape[0]
            ptr = int(self.ptr[k].item())
            if nk >= self.size:
                self.queue[k] = fk[:self.size]
                self.ptr[k] = 0
                self.filled[k] = True
            else:
                end = ptr + nk
                if end <= self.size:
                    self.queue[k, ptr:end] = fk
                else:
                    first = self.size - ptr
                    self.queue[k, ptr:] = fk[:first]
                    self.queue[k, :nk - first] = fk[first:]
                self.ptr[k] = end % self.size
                if end >= self.size:
                    self.filled[k] = True

    def all_features_labels(self):
        """Concatenated features + labels for every class with any entry."""
        feats, labs = [], []
        for k in range(self.num_classes):
            n = self.size if bool(self.filled[k].item()) else int(self.ptr[k].item())
            if n > 0:
                feats.append(self.queue[k, :n])
                labs.append(torch.full((n,), k, device=self.device, dtype=torch.long))
        if not feats:
            return None, None
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


def sample_anchors_from_labeled(features, labels, pred,
                                max_per_class=64, ignore_index=255,
                                bank_filter='clean', conf=None, conf_thresh=0.95):
    """Class-balanced anchor pool from a labeled batch, under one of three
    admission rules (the paper's ablation / decomposition axis).

    Args:
        features: [B, D, H, W] L2-normalised embeddings.
        labels: [B, H, W] ground-truth class indices (use ignore_index for void).
        pred: [B, H, W] argmax of student logits at the SAME resolution.
        max_per_class: cap per-class anchors for class-balanced sampling.
        bank_filter: admission rule, one of
            'clean'   (default, PixCon): labeled AND pred == label. Gives
                      rho_F = 0 AND the correctness / g_T-sharpening effect.
            'labeled' : labeled only (ignore the prediction). Gives rho_F = 0
                      WITHOUT correctness, so contrasting it against 'clean'
                      isolates the correctness lever, and against 'conf'
                      isolates the rho_F = 0 lever (reviewer decomposition).
            'conf'    : ReCo/U^2PL-style. labeled AND conf >= conf_thresh, with
                      the *predicted* class as the bank label (so rho_F > 0).
                      Requires `conf`.
        conf: [B, H, W] max-softmax confidence, required iff bank_filter == 'conf'.
        conf_thresh: confidence threshold for the 'conf' rule.

    Returns:
        (anchor_feats [N_a, D], anchor_labels [N_a]).
    """
    b, d, _, _ = features.shape
    feats_flat = features.permute(0, 2, 3, 1).reshape(-1, d)
    labels_flat = labels.reshape(-1)
    pred_flat = pred.reshape(-1)
    if bank_filter == 'conf':
        if conf is None:
            raise ValueError("bank_filter='conf' requires the per-pixel `conf` tensor")
        valid = (labels_flat != ignore_index) & (conf.reshape(-1) >= conf_thresh)
        label_src = pred_flat            # predicted class -> bank can be wrong (rho_F > 0)
    elif bank_filter == 'labeled':
        valid = (labels_flat != ignore_index)
        label_src = labels_flat          # ground-truth label -> rho_F = 0, no correctness
    else:  # 'clean' (PixCon default)
        valid = (labels_flat != ignore_index) & (pred_flat == labels_flat)
        label_src = labels_flat          # ground-truth label -> rho_F = 0 + correctness
    if not valid.any():
        return features.new_zeros((0, d)), labels.new_zeros((0,), dtype=torch.long)
    fv = feats_flat[valid]; lv = label_src[valid]
    keep_f, keep_l = [], []
    for k in lv.unique():
        m = lv == k
        n = int(m.sum().item())
        if n <= max_per_class:
            keep_f.append(fv[m]); keep_l.append(lv[m])
        else:
            idx = torch.randperm(n, device=fv.device)[:max_per_class]
            keep_f.append(fv[m][idx]); keep_l.append(lv[m][idx])
    return torch.cat(keep_f, dim=0), torch.cat(keep_l, dim=0)


def pixcon_loss(anchor_feats, anchor_labels, bank, temperature=0.1, max_anchors=1024):
    """Supervised InfoNCE against the clean-positive bank.

    Returns zero if the bank is empty or no anchor has same-class positives.
    """
    n = anchor_feats.shape[0]
    if n == 0:
        return anchor_feats.new_zeros(())
    if n > max_anchors:
        idx = torch.randperm(n, device=anchor_feats.device)[:max_anchors]
        anchor_feats = anchor_feats[idx]; anchor_labels = anchor_labels[idx]
    bf, bl = bank.all_features_labels()
    if bf is None or bf.shape[0] == 0:
        return anchor_feats.new_zeros(())
    logits = anchor_feats @ bf.t() / temperature                       # [N_a, N_b]
    same = anchor_labels.unsqueeze(1) == bl.unsqueeze(0)
    max_l = logits.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits - max_l)
    sum_all = exp.sum(dim=1)
    sum_pos = (exp * same.float()).sum(dim=1)
    has_pos = sum_pos > 0
    if not has_pos.any():
        return anchor_feats.new_zeros(())
    return -(torch.log(sum_pos[has_pos] + 1e-9) - torch.log(sum_all[has_pos] + 1e-9)).mean()
