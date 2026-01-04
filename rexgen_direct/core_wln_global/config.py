# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: config.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 18:37
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: config.py
"""
Configuration file for the direct core finder.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    hidden_size: int = 300
    depth: int = 3
    batch_size: int = 2
    atom_fdim: Optional[int] = None
    bond_fdim: Optional[int] = None
    max_nb: int = 10
    use_rich_features: bool = False

    # Training
    learning_rate: float = 0.001
    max_norm: float = 5.0
    num_epochs: int = 100

    # Inference
    top_k: int = 80
    beam_size: int = 12

    # Paths
    model_dir: str = "models"
    data_dir: str = "data"

    def __post_init__(self):
        """Set default feature dimensions."""
        if self.atom_fdim is None:
            if self.use_rich_features:
                # Will be determined dynamically
                pass
            else:
                from rexgen_direct.core_wln_global.mol_graph import atom_fdim as default_adim
                self.atom_fdim = default_adim

        if self.bond_fdim is None:
            from rexgen_direct.core_wln_global.mol_graph import bond_fdim as default_bdim
            self.bond_fdim = default_bdim


# Default configurations
DEFAULT_CONFIG = ModelConfig()
RICH_CONFIG = ModelConfig(use_rich_features=True)

# Model paths
MODEL_PATHS = {
    'default': os.path.join(os.path.dirname(__file__),
                            "model-300-3-direct", "model.ckpt-140000"),
    'rich': os.path.join(os.path.dirname(__file__),
                         "model-300-3-direct-rich", "model.ckpt-140000")
}
