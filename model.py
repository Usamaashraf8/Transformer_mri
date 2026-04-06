"""
Transformer image classifier built on top of torchvision backbones.
"""

import torch.nn as nn
from torchvision.models import (
    Swin_T_Weights,
    ViT_B_16_Weights,
    swin_t,
    vit_b_16,
)

import config


def _build_swin_t():
    weights = Swin_T_Weights.IMAGENET1K_V1 if config.USE_PRETRAINED else None
    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=config.HEAD_DROPOUT),
        nn.Linear(in_features, config.NUM_CLASSES),
    )
    return model


def _build_vit_b_16():
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if config.USE_PRETRAINED else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=config.HEAD_DROPOUT),
        nn.Linear(in_features, config.NUM_CLASSES),
    )
    return model


def _freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    if config.MODEL_NAME == "swin_t":
        for param in model.head.parameters():
            param.requires_grad = True
    elif config.MODEL_NAME == "vit_b_16":
        for param in model.heads.head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model name: {config.MODEL_NAME}")


def unfreeze_backbone(model):
    if config.MODEL_NAME == "swin_t":
        trainable_prefixes = ("features.6", "features.7", "norm", "permute", "avgpool", "head")
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
        return

    for param in model.parameters():
        param.requires_grad = True


def build_model():
    if config.MODEL_NAME == "swin_t":
        model = _build_swin_t()
    elif config.MODEL_NAME == "vit_b_16":
        model = _build_vit_b_16()
    else:
        raise ValueError(f"Unsupported model name: {config.MODEL_NAME}")

    if config.USE_PRETRAINED and config.FREEZE_BACKBONE_EPOCHS > 0:
        _freeze_backbone(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model: {config.MODEL_NAME} | "
        f"Total params: {total_params:,} | Trainable params: {trainable_params:,}"
    )
    return model
