"""Classification heads for cube MLP models.

Three pooling strategies:
- PoolHead: mean-pool over positions (loses spatial info — avoid for equivariance tests)
- PerPositionHead: per-position classification then vote (preserves spatial info)
- AttentionHead: learned position weights (best for rotation-robust classification)
"""

import torch
import torch.nn as nn


class PoolHead(nn.Module):
    """Mean-pool over positions → linear classifier.

    NOTE: This destroys spatial information. Do NOT use for rotation
    equivariance experiments — rotated and un-rotated inputs produce
    identical pooled representations.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.mean(dim=1))


class PerPositionHead(nn.Module):
    """Per-position classification → mean vote.

    Each position independently predicts the class. The final prediction
    is the average over all positions, making it rotation-INVARIANT.

    Use for tasks where the final classification should be invariant to
    permutations but the backbone should be equivariant.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).mean(dim=1)


class PerPositionNoPoolHead(nn.Module):
    """Per-position classification WITHOUT pooling.

    Returns [B, N, n_classes] — each position independently predicts.
    Use this for position reconstruction and other tasks where the
    output MUST be per-position.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  # [B, N, n_classes]


class AttentionHead(nn.Module):
    """Learned attention over positions → classifier.

    The attention weights can learn to "look at" specific spatial
    patterns regardless of their position. This provides rotation
    invariance through learning rather than through pooling.

    Best choice for rotation-robust classification tasks.
    """
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.attention(x), dim=1)  # [B, N, 1]
        pooled = (x * attn).sum(dim=1)                    # [B, D]
        return self.classifier(pooled)


class ModelWrapper(nn.Module):
    """Wraps backbone + classification head into one module."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
