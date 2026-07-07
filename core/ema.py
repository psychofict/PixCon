"""EMA teacher with ramp-up decay schedule (UniMatch V2-style)."""

from copy import deepcopy

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """Frozen mirror of the student, updated by an EMA of the student's weights.

    Updates parameters AND floating-point buffers (e.g. BN running stats);
    non-float buffers are copied directly. Use teacher.update(student) after
    each optimizer step; set teacher.decay externally with the ramp-up schedule
    `min(1 - 1/(iter+1), 0.996)`.
    """

    def __init__(self, student, decay=0.996):
        super().__init__()
        self.decay = decay
        self.model = deepcopy(student)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student):
        d = self.decay
        sp = dict(student.named_parameters())
        for name, p in self.model.named_parameters():
            if name in sp:
                p.data.mul_(d).add_(sp[name].data, alpha=1.0 - d)
        sb = dict(student.named_buffers())
        for name, b in self.model.named_buffers():
            if name in sb:
                if b.dtype.is_floating_point:
                    b.data.mul_(d).add_(sb[name].data, alpha=1.0 - d)
                else:
                    b.data.copy_(sb[name].data)

    def forward(self, x):
        return self.model(x)

    def state_dict(self, **kw):
        return {'decay': self.decay, 'model_state_dict': self.model.state_dict()}

    def load_state_dict(self, sd, **kw):
        self.decay = sd['decay']
        self.model.load_state_dict(sd['model_state_dict'])
