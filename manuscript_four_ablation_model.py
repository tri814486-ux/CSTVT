"""Single-module ablations of CSTVT.

Each variant is built from a complete model and then has exactly one component
removed, so all retained parameters are element-wise identical at the same seed.
Every variant keeps the Class Token readout and the block ordering.

Note on the readout: the last block updates its Class Token before its ConvFFN,
and the classifier reads the final Class Token. The last block's visual ConvFFN
output therefore does not reach the logits, while earlier blocks' ConvFFN
outputs still influence them through the subsequent blocks.
"""
from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

try:
    from .manuscript_primary_model import ManuscriptCSTVT, ManuscriptSTBlock
except ImportError:
    from manuscript_primary_model import ManuscriptCSTVT, ManuscriptSTBlock


ABLATION_METADATA = {
    "full": {
        "label": "CSTVT",
        "removals": [],
        "retained": ["CNN embedding", "Class Token", "absolute position P",
                     "CPE", "STS", "TU", "MHSA", "ConvFFN"],
    },
    "vit": {
        "label": "CNN-ViT",
        "removals": ["STS", "TU association remapping"],
        "replacement": "MHSA over Class Token and all normalized visual tokens",
        "retained": ["CNN embedding", "Class Token", "absolute position P",
                     "CPE", "MHSA", "ConvFFN"],
    },
    "no_cpe": {
        "label": "CSTVT w/o CPE",
        "removals": ["block CPE depthwise convolution and its residual branch"],
        "retained": ["CNN embedding", "Class Token", "absolute position P",
                     "STS", "TU", "MHSA", "ConvFFN"],
    },
    "no_dwconv": {
        "label": "CSTVT w/o DWConv",
        "removals": ["ConvFFN 3x3 depthwise convolution only"],
        "retained": ["CNN embedding", "Class Token", "absolute position P",
                     "CPE", "STS", "TU", "MHSA", "ConvFFN BN/1x1/GELU/1x1"],
    },
}


class _IsolatedAblationBlock(ManuscriptSTBlock):
    """Reuse existing children without creating or reinitializing shared weights."""

    def __init__(self, original, *, direct_visual_attention=False, remove_cpe=False):
        nn.Module.__init__(self)
        self.direct_visual_attention = bool(direct_visual_attention)
        self.remove_cpe = bool(remove_cpe)
        self.cpe = None if self.remove_cpe else original.cpe
        self.norm = original.norm
        self.sts = None if self.direct_visual_attention else original.sts
        self.mhsa = original.mhsa
        self.ffn = original.ffn
        self.train(original.training)

    def forward(self, features, class_token):
        # A removed CPE contributes zero; Identity inside features + CPE(features)
        # would incorrectly double the input.
        position_enhanced = features if self.remove_cpe else features + self.cpe(features)
        normalized = self.norm(position_enhanced.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.direct_visual_attention:
            visual_tokens = normalized.flatten(2).transpose(1, 2)
            enhanced = self.mhsa(torch.cat([class_token, visual_tokens], dim=1))
            remapped = enhanced[:, 1:].transpose(1, 2).reshape_as(position_enhanced)
        else:
            super_tokens, association = self.sts(normalized)
            enhanced = self.mhsa(torch.cat([class_token, super_tokens], dim=1))
            remapped = association.to(enhanced.dtype) @ enhanced[:, 1:]
            remapped = remapped.transpose(1, 2).reshape_as(position_enhanced)
        class_token = class_token + enhanced[:, :1]
        visual = position_enhanced + remapped
        return visual + self.ffn(visual), class_token


def build_manuscript_ablation(key="full", *, seed=None, **kwargs):
    """Build one variant; kwargs are identical to ManuscriptCSTVT kwargs.

    ``seed`` is optional. When provided, construction uses a temporary CPU RNG
    state and leaves the caller's RNG state unchanged. No CUDA work is done here.
    Without it, callers can use the same torch.manual_seed before each build.
    """
    if key not in ABLATION_METADATA:
        raise ValueError(f"Unknown ablation {key!r}; choose {tuple(ABLATION_METADATA)}")
    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.default_generator.manual_seed(int(seed))
        model = ManuscriptCSTVT(**kwargs)

    if key in ("vit", "no_cpe"):
        model.blocks = nn.ModuleList([
            _IsolatedAblationBlock(block,
                                   direct_visual_attention=key == "vit",
                                   remove_cpe=key == "no_cpe")
            for block in model.blocks
        ])
    elif key == "no_dwconv":
        for block in model.blocks:
            block.ffn[3] = nn.Identity()

    model.ablation_key = key
    model.ablation_metadata = deepcopy(ABLATION_METADATA[key])
    model.ablation_metadata["inherited_readout_note"] = (
        "Last-block ConvFFN output is not consumed by the Class Token classifier; "
        "the original forward ordering is retained in every variant."
    )
    model.settings["ablation"] = key
    return model


__all__ = ["build_manuscript_ablation", "ABLATION_METADATA"]
