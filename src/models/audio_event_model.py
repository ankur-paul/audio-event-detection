"""
Model architecture for multi-label audio event detection.

Supports:
- EfficientNet-B0/B2/etc. as CNN backbone
- Frame-level predictions (weakly-supervised SED)
- Multiple temporal pooling strategies (max, mean, attention)
- Both clip-level and frame-level output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()


class AttentionPooling(nn.Module):
    """
    Attention-based temporal pooling.

    Learns attention weights over time frames to produce clip-level predictions
    from frame-level predictions.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1),  # softmax over time dimension
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Frame-level features of shape (batch, time, features).

        Returns:
            Tuple of:
                - Pooled output of shape (batch, features)
                - Attention weights of shape (batch, time, num_classes)
        """
        attn_weights = self.attention(x)  # (batch, time, num_classes)
        # Weighted sum over time
        # x: (batch, time, features), attn: (batch, time, num_classes)
        # We need to handle this differently since features != num_classes
        return attn_weights


class AudioEventDetectionModel(nn.Module):
    """
    Multi-label audio event detection model with frame-level predictions.

    Architecture:
        Log-Mel Spectrogram (1, n_mels, time)
        → EfficientNet backbone (feature extraction)
        → Temporal feature maps
        → Frame-level classifier (1D conv / linear)
        → Temporal pooling (max / mean / attention)
        → Clip-level predictions (sigmoid)

    The model outputs both frame-level and clip-level predictions.
    Training uses clip-level loss (weak supervision).
    Inference uses frame-level predictions for event timeline.
    """

    def __init__(
        self,
        num_classes: int = 50,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        pooling: str = "attention",
        frame_level: bool = True,
    ):
        """
        Args:
            num_classes: Number of sound event classes.
            backbone: Backbone model name.
            pretrained: Whether to use pretrained weights.
            dropout: Dropout probability.
            pooling: Temporal pooling method ('max', 'mean', 'attention').
            frame_level: Whether to produce frame-level predictions.
        """
        super().__init__()

        self.num_classes = num_classes
        self.pooling_type = pooling
        self.frame_level = frame_level

        # Build backbone
        self.backbone, self.backbone_features = self._build_backbone(
            backbone, pretrained
        )

        # Frame-level classifier
        self.dropout = nn.Dropout(p=dropout)
        self.frame_classifier = nn.Conv1d(
            in_channels=self.backbone_features,
            out_channels=num_classes,
            kernel_size=1,
            bias=True,
        )

        # Attention pooling (optional)
        if pooling == "attention":
            self.attention_fc = nn.Sequential(
                nn.Linear(self.backbone_features, self.backbone_features // 4),
                nn.ReLU(),
                nn.Linear(self.backbone_features // 4, 1),
            )

        logger.info(
            f"Model: backbone={backbone}, classes={num_classes}, "
            f"pooling={pooling}, frame_level={frame_level}, "
            f"backbone_features={self.backbone_features}"
        )

    def _build_backbone(
        self, backbone_name: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """
        Build the CNN backbone.

        Removes the classification head and global pooling from EfficientNet,
        keeping only the feature extraction layers.

        Returns:
            Tuple of (backbone_module, num_features).
        """
        try:
            import timm

            model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=1,  # single-channel spectrogram input
                features_only=False,
                num_classes=0,  # remove classification head
                global_pool="",  # remove global pooling - we need spatial dims
            )
            num_features = model.num_features

        except ImportError:
            raise ImportError(
                "timm is required for EfficientNet backbone. "
                "Install with: pip install timm"
            )

        return model, num_features

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input spectrogram of shape (batch, 1, n_mels, time_frames).

        Returns:
            Dictionary with:
                - 'clip_logits': Clip-level logits (batch, num_classes)
                - 'clip_probs': Clip-level probabilities (batch, num_classes)
                - 'frame_logits': Frame-level logits (batch, num_classes, time) [if frame_level]
                - 'frame_probs': Frame-level probabilities (batch, num_classes, time) [if frame_level]
                - 'attention_weights': Attention weights (batch, time) [if attention pooling]
        """
        outputs = {}

        # Backbone feature extraction
        # x: (batch, 1, n_mels, time_frames) → features: (batch, C, H, W)
        features = self.backbone(x)

        # If features have spatial dimensions (H, W), collapse frequency dim
        if features.dim() == 4:
            # (batch, channels, freq, time) → (batch, channels, time)
            features = features.mean(dim=2)
        elif features.dim() == 3:
            pass  # already (batch, channels, time)
        elif features.dim() == 2:
            # (batch, channels) - no time dimension preserved
            features = features.unsqueeze(2)  # (batch, channels, 1)

        # features: (batch, backbone_features, time_steps)
        features = self.dropout(features)

        # Frame-level predictions via 1D convolution over time
        frame_logits = self.frame_classifier(features)  # (batch, num_classes, time)

        if self.frame_level:
            outputs["frame_logits"] = frame_logits
            outputs["frame_probs"] = torch.sigmoid(frame_logits)

        # Temporal pooling to get clip-level predictions
        if self.pooling_type == "max":
            clip_logits, _ = frame_logits.max(dim=2)  # (batch, num_classes)

        elif self.pooling_type == "mean":
            clip_logits = frame_logits.mean(dim=2)  # (batch, num_classes)

        elif self.pooling_type == "attention":
            # features: (batch, backbone_features, time) → (batch, time, backbone_features)
            feat_permuted = features.permute(0, 2, 1)
            # Compute attention weights over time
            attn_logits = self.attention_fc(feat_permuted).squeeze(-1)  # (batch, time)
            attn_weights = F.softmax(attn_logits, dim=1)  # (batch, time)
            outputs["attention_weights"] = attn_weights

            # Weighted average of frame logits
            # frame_logits: (batch, num_classes, time)
            # attn_weights: (batch, 1, time)
            attn_weights_expanded = attn_weights.unsqueeze(1)
            clip_logits = (frame_logits * attn_weights_expanded).sum(dim=2)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        outputs["clip_logits"] = clip_logits
        outputs["clip_probs"] = torch.sigmoid(clip_logits)

        return outputs


def build_model(config) -> AudioEventDetectionModel:
    """
    Build model from configuration.

    Args:
        config: Full configuration object.

    Returns:
        AudioEventDetectionModel instance.
    """
    model_cfg = config.model
    model = AudioEventDetectionModel(
        num_classes=model_cfg.num_classes,
        backbone=model_cfg.backbone,
        pretrained=model_cfg.pretrained,
        dropout=model_cfg.dropout,
        pooling=model_cfg.pooling,
        frame_level=model_cfg.frame_level,
    )
    return model
