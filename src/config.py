"""Load config.yaml and resolve the single-vs-multi-checkpoint issue.

All three original notebooks read their paths from hardcoded strings
scattered through the code (and, worse, from *different* Kaggle-uploaded
checkpoints for the VAE/DINO rows vs the YOLOv8 baseline). This module is
the one place every script pulls paths from.
"""
import os
import torch
import yaml


def _resolve(value, root):
    """Resolve simple ${a.b} references against the root config dict."""
    if isinstance(value, str) and "${" in value:
        while "${" in value:
            start = value.index("${")
            end = value.index("}", start)
            ref = value[start + 2 : end]
            node = root
            for part in ref.split("."):
                node = node[part]
            value = value[:start] + str(node) + value[end + 1 :]
        return value
    return value


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                section[k] = _resolve(v, cfg)

    return cfg


def get_device(cfg):
    requested = cfg.get("eval", {}).get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
