# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: directcorefinder.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:14
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: directcorefinder.py
"""
DirectCoreFinder class for deploying the core finding model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from typing import List, Tuple, Optional, Dict, Any
import rdkit.Chem as Chem
from tqdm import tqdm

from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from rexgen_direct.core_wln_global.ioutils_direct import get_all_batch, INVALID_BOND, binary_fdim
from functools import partial

NK3 = 80
batch_size = 2  # just fake it, make two
hidden_size = 300
depth = 3
model_path = os.path.join(os.path.dirname(__file__), "model-300-3-direct/model.ckpt-140000")

# Create partial function for batch processing
smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)


class DirectCoreFinderModel(nn.Module):
    """PyTorch model for direct core finding."""

    def __init__(self, hidden_size: int = hidden_size, depth: int = depth,
                 atom_fdim: int = None, bond_fdim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # 如果未提供维度，使用默认值
        if atom_fdim is None:
            from rexgen_direct.core_wln_global.mol_graph import atom_fdim as default_adim
            atom_fdim = default_adim
        if bond_fdim is None:
            from rexgen_direct.core_wln_global.mol_graph import bond_fdim as default_bdim
            bond_fdim = default_bdim

        # 直接创建 RCNN_WL_Last 模块
        from rexgen_direct.core_wln_global.models import RCNN_WL_Last
        self.rcnn = RCNN_WL_Last(hidden_size, depth, atom_fdim)

        # Attention layers
        self.att_atom_feature = nn.Linear(hidden_size, hidden_size, bias=False)
        self.att_bin_feature = nn.Linear(binary_fdim, hidden_size, bias=False)
        self.att_scores = nn.Linear(hidden_size, 1)

        # Pair processing layers
        self.atom_feature = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bin_feature = nn.Linear(binary_fdim, hidden_size, bias=False)
        self.ctx_feature = nn.Linear(hidden_size, hidden_size, bias=False)

        # Final scoring layer
        self.scores = nn.Linear(hidden_size, 5)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize linear layers
        for module in [self.att_atom_feature, self.att_bin_feature,
                       self.atom_feature, self.bin_feature, self.ctx_feature]:
            nn.init.normal_(module.weight, std=0.1)

        nn.init.normal_(self.att_scores.weight, std=0.1)
        if self.att_scores.bias is not None:
            nn.init.constant_(self.att_scores.bias, 0.0)

        nn.init.normal_(self.scores.weight, std=0.1)
        if self.scores.bias is not None:
            nn.init.constant_(self.scores.bias, 0.0)

    def forward(self, graph_inputs: Tuple[torch.Tensor, ...],
                binary: torch.Tensor, label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        """
        device = next(self.parameters()).device

        # 将输入移动到正确设备
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = graph_inputs

        input_atom = input_atom.to(device)
        input_bond = input_bond.to(device)
        atom_graph = atom_graph.to(device)
        bond_graph = bond_graph.to(device)
        num_nbs = num_nbs.to(device)
        node_mask = node_mask.to(device)

        binary = binary.to(device)
        if label is not None:
            label = label.to(device)

        # 重新构建graph_inputs
        graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)

        batch_size = input_atom.shape[0]

        # 使用 self.rcnn 而不是 lambda 函数
        atom_hiddens, _ = self.rcnn(graph_inputs, batch_size)

        # Calculate local atom pair features
        atom_hiddens1 = atom_hiddens.unsqueeze(1)  # [batch, 1, atoms, hidden]
        atom_hiddens2 = atom_hiddens.unsqueeze(2)  # [batch, atoms, 1, hidden]
        atom_pair = atom_hiddens1 + atom_hiddens2  # [batch, atoms, atoms, hidden]

        # Calculate attention scores
        att_hidden = F.relu(self.att_atom_feature(atom_pair) + self.att_bin_feature(binary))
        att_score = torch.sigmoid(self.att_scores(att_hidden))  # [batch, atoms, atoms, 1]

        # Calculate context features
        att_context = att_score * atom_hiddens1
        att_context = torch.sum(att_context, dim=2)  # [batch, atoms, hidden]

        # Calculate global atom pair features
        att_context1 = att_context.unsqueeze(1)  # [batch, 1, atoms, hidden]
        att_context2 = att_context.unsqueeze(2)  # [batch, atoms, 1, hidden]
        att_pair = att_context1 + att_context2  # [batch, atoms, atoms, hidden]

        # Calculate pair hidden features
        pair_hidden = (self.atom_feature(atom_pair) +
                       self.bin_feature(binary) +
                       self.ctx_feature(att_pair))
        pair_hidden = F.relu(pair_hidden)

        # Reshape for scoring
        batch_size, num_atoms, _, hidden = pair_hidden.shape
        pair_hidden_flat = pair_hidden.reshape(batch_size, -1, hidden)

        # Calculate scores
        score = self.scores(pair_hidden_flat)  # [batch, atoms*atoms, 5]
        score = score.reshape(batch_size, -1)  # [batch, atoms*atoms*5]

        # Prepare outputs
        outputs = {
            'score': score,
            'att_score': att_score.squeeze(-1)
        }

        # If labels are provided, calculate top-k predictions
        if label is not None:
            # Create mask for invalid bonds
            bmask = (label == INVALID_BOND).float() * 10000.0
            score_masked = score - bmask

            # Get top-k predictions
            topk_scores, topk = torch.topk(score_masked, k=NK3, dim=1)
            outputs.update({
                'topk': topk,
                'topk_scores': topk_scores,
                'label_dim': torch.tensor(score.shape[1], dtype=torch.int32)
            })

        return outputs


class DirectCoreFinder:
    """Main class for core finding inference."""

    def __init__(self, hidden_size: int = hidden_size,
                 batch_size: int = batch_size, depth: int = depth):
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.depth = depth
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: str = model_path):
        """Load the trained model."""
        print(f"Loading model from {model_path}...")

        # Initialize model
        self.model = DirectCoreFinderModel(
            hidden_size=self.hidden_size,
            depth=self.depth
        )

        # Load weights
        if os.path.exists(model_path):
            try:
                # Try loading PyTorch weights
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print("PyTorch weights loaded successfully.")
            except:
                print("Failed to load PyTorch weights, initializing with random weights.")
        else:
            print(f"Model file not found at {model_path}, using random initialization.")

        # Move to device
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on device: {self.device}")

    def predict(self, reactants_smi: str) -> Tuple[str, List[str], List[float], np.ndarray]:
        """
        Predict bond changes for given reactants.

        Args:
            reactants_smi: SMILES string of reactants

        Returns:
            Tuple of (canonical_smiles, bond_predictions, bond_scores, attention_scores)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Bond order mappings
        bo_to_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}
        bindex_to_o = {val: key for key, val in bo_to_index.items()}
        nbos = len(bo_to_index)

        # Prepare input
        m = Chem.MolFromSmiles(reactants_smi)
        if m is None:
            raise ValueError(f"Invalid SMILES string: {reactants_smi}")

        # Add atom mapping if not present
        if any(not a.HasProp('molAtomMapNumber') for a in m.GetAtoms()):
            mapnum = 1
            for a in m.GetAtoms():
                a.SetIntProp('molAtomMapNumber', mapnum)
                mapnum += 1

        react = Chem.MolToSmiles(m)

        # Create batch (duplicate for batch processing)
        src_batch = [react, react]
        edit_batch = ['0-1-0.0', '0-1-0.0']  # dummy edits

        # Prepare graph inputs
        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(list(zip(src_batch, edit_batch)))

        # Convert to PyTorch tensors
        src_tensors = []
        for arr in src_tuple:
            tensor = torch.from_numpy(arr).float().to(self.device)
            src_tensors.append(tensor)

        binary_tensor = torch.from_numpy(cur_bin).float().to(self.device)
        label_tensor = torch.from_numpy(cur_label).long().to(self.device)

        # Run prediction
        with torch.no_grad():
            outputs = self.model(tuple(src_tensors), binary_tensor, label_tensor)

            cur_topk = outputs['topk'][0].cpu().numpy()
            cur_sco = outputs['topk_scores'][0].cpu().numpy()
            cur_att_score = outputs['att_score'][0].cpu().numpy()

            # Get label dimension
            score_shape = outputs['score'].shape[1]
            cur_dim = int(math.sqrt(score_shape / 5))  # important! changed to divide by 5

        # Process predictions
        bond_preds = []
        bond_scores = []

        for j in range(NK3):
            k = cur_topk[j]
            bindex = k % nbos
            y = ((k - bindex) // nbos) % cur_dim + 1
            x = (k - bindex - (y - 1) * nbos) // (cur_dim * nbos) + 1

            if x < y:  # keep canonical
                bo = bindex_to_o[bindex]
                bond_preds.append(f"{x}-{y}-{bo:.1f}")
                bond_scores.append(cur_sco[j])

        return react, bond_preds, bond_scores, cur_att_score

    def predict_batch(self, reactants_list: List[str],
                      show_progress: bool = True) -> List[Tuple[str, List[str], List[float], np.ndarray]]:
        """
        Predict bond changes for a batch of reactants.

        Args:
            reactants_list: List of SMILES strings
            show_progress: Whether to show progress bar

        Returns:
            List of prediction tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = []

        if show_progress:
            iterator = tqdm(reactants_list, desc="Predicting")
        else:
            iterator = reactants_list

        for reactants_smi in iterator:
            try:
                result = self.predict(reactants_smi)
                results.append(result)
            except Exception as e:
                print(f"Error predicting for {reactants_smi}: {e}")
                results.append((reactants_smi, [], [], np.array([])))

        return results


if __name__ == '__main__':
    # Test the module
    directcorefinder = DirectCoreFinder()
    directcorefinder.load_model()

    # Test reaction
    react = '[F:1][C:2]([C:3](=[C:4]([F:5])[F:6])[F:7])([F:8])[F:9].[H:10][H:11]'

    print(f"Predicting for reaction: {react}")
    res = directcorefinder.predict(react)

    print(f"\nCanonical SMILES: {res[0]}")
    print(f"\nTop bond predictions:")
    for i, (pred, score) in enumerate(zip(res[1][:10], res[2][:10])):
        print(f"  {i + 1}. {pred} (score: {score:.3f})")

    print(f"\nAttention scores shape: {res[3].shape}")
