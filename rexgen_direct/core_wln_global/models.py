# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: models.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:13
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from rexgen_direct.core_wln_global.nn import LinearND, Linear


class RCNN_WL_Last(nn.Module):
    """
    WLN embedding module (local, no attention mechanism).

    This performs the Weisfeiler-Lehman neural network embedding.
    """

    def __init__(self, hidden_size: int, depth: int, atom_fdim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # Atom embedding - 需要传入输入维度
        self.atom_embedding = LinearND(atom_fdim, hidden_size)

        # WL layers
        self.wl_layers = nn.ModuleList()
        for i in range(depth):
            layer = WL_Layer(hidden_size, bond_fdim=6)  # bond_fdim 需要传入
            self.wl_layers.append(layer)

    def forward(self, graph_inputs: Tuple[torch.Tensor, ...], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)

        # Initial atom embedding
        atom_features = F.relu(self.atom_embedding(input_atom))

        # Store intermediate layers
        layers = []

        # Apply WL layers
        for i, wl_layer in enumerate(self.wl_layers):
            atom_features, layer_output = wl_layer(
                atom_features, input_bond, atom_graph, bond_graph,
                num_nbs, node_mask, batch_size
            )
            layers.append(layer_output)

        # Use final layer output as atom fingerprints
        kernels = layers[-1]  # atom FPs are the final output after "depth" convolutions

        # Molecular FP is sum over atom FPs
        fp = torch.sum(kernels, dim=1)

        return kernels, fp


# 修改 WL_Layer 类的 __init__ 方法
class WL_Layer(nn.Module):
    """Single WL convolution layer."""

    def __init__(self, hidden_size: int, bond_fdim: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Neighbor aggregation
        self.nei_atom = LinearND(hidden_size, hidden_size)
        self.nei_bond = LinearND(bond_fdim, hidden_size)

        # Self transformation
        self.self_atom = LinearND(hidden_size, hidden_size)

        # Label update
        self.label_U2 = LinearND(hidden_size + bond_fdim, hidden_size)
        self.label_U1 = LinearND(hidden_size * 2, hidden_size)

    def forward(self, atom_features: torch.Tensor, input_bond: torch.Tensor,
                atom_graph: torch.Tensor, bond_graph: torch.Tensor,
                num_nbs: torch.Tensor, node_mask: torch.Tensor,
                batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for WL layer.
        """
        from rexgen_direct.core_wln_global.mol_graph import max_nb

        # Gather neighbor atom features
        fatom_nei = self._gather_nd(atom_features, atom_graph)  # [batch, atoms, max_nb, hidden]

        # Gather neighbor bond features
        fbond_nei = self._gather_nd(input_bond, bond_graph)  # [batch, atoms, max_nb, bond_fdim]

        # Process neighbor information
        h_nei_atom = self.nei_atom(fatom_nei)
        h_nei_bond = self.nei_bond(fbond_nei)
        h_nei = h_nei_atom * h_nei_bond

        # Create neighbor mask
        mask_nei = self._create_neighbor_mask(num_nbs, batch_size, atom_features.shape[1])
        f_nei = torch.sum(h_nei * mask_nei, dim=2)

        # Self transformation
        f_self = self.self_atom(atom_features)

        node_mask = node_mask.unsqueeze(-1)
        # Layer output
        layer_output = f_nei * f_self * node_mask

        # Update atom labels
        l_nei = torch.cat([fatom_nei, fbond_nei], dim=3)
        nei_label = F.relu(self.label_U2(l_nei))
        nei_label = torch.sum(nei_label * mask_nei, dim=2)

        new_label = torch.cat([atom_features, nei_label], dim=2)
        new_label = self.label_U1(new_label)
        updated_features = F.relu(new_label)

        return updated_features, layer_output

    def _gather_nd(self, params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        PyTorch implementation of tf.gather_nd.

        Args:
            params: Tensor to gather from
            indices: Tensor of indices with shape [..., 2]

        Returns:
            Gathered tensor
        """
        # indices shape: [batch, atoms, max_nb, 2]
        batch_indices = indices[..., 0].long()  # Batch indices
        elem_indices = indices[..., 1].long()  # Element indices

        # Create linear indices for 2D gather
        batch_size, num_atoms, max_nb = indices.shape[:3]

        # Reshape for easier indexing
        params_flat = params.reshape(-1, params.shape[-1])

        # Calculate linear indices
        batch_offset = torch.arange(batch_size, device=params.device).reshape(-1, 1, 1) * params.shape[1]
        linear_indices = (batch_offset + elem_indices).reshape(-1)

        # Gather
        gathered = params_flat[linear_indices]

        # Reshape back
        return gathered.reshape(batch_size, num_atoms, max_nb, -1)

    def _create_neighbor_mask(self, num_nbs: torch.Tensor, batch_size: int,
                              num_atoms: int) -> torch.Tensor:
        """
        Create mask for neighbor aggregation.
        """
        from rexgen_direct.core_wln_global.mol_graph import max_nb

        # Reshape num_nbs and create sequence mask
        num_nbs_flat = num_nbs.reshape(-1)

        # Create mask for each atom's neighbors
        max_nb_tensor = max_nb
        range_tensor = torch.arange(max_nb_tensor, device=num_nbs.device).unsqueeze(0)

        # Compare each position with num_nbs
        mask = (range_tensor < num_nbs_flat.unsqueeze(1)).float()

        # Reshape back to [batch_size, num_atoms, max_nb, 1]
        mask = mask.reshape(batch_size, num_atoms, max_nb_tensor, 1)

        return mask


def rcnn_wl_last(graph_inputs: Tuple[torch.Tensor, ...], batch_size: int,
                 hidden_size: int, depth: int, training: bool = True,
                 atom_fdim: int = None, bond_fdim: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional version of RCNN_WL_Last for backward compatibility.
    """
    if atom_fdim is None:
        # 从输入获取 atom_fdim
        input_atom = graph_inputs[0]
        atom_fdim = input_atom.shape[-1]

    model = RCNN_WL_Last(hidden_size, depth, atom_fdim)
    if training:
        model.train()
    else:
        model.eval()

    return model(graph_inputs, batch_size)


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    num_atoms = 10
    atom_fdim = 100
    bond_fdim = 6

    # Create dummy inputs
    input_atom = torch.randn(batch_size, num_atoms, atom_fdim)
    input_bond = torch.randn(batch_size, num_atoms, bond_fdim)
    atom_graph = torch.randint(0, num_atoms, (batch_size, num_atoms, 10, 2))
    bond_graph = torch.randint(0, num_atoms, (batch_size, num_atoms, 10, 2))
    num_nbs = torch.randint(1, 5, (batch_size, num_atoms))
    node_mask = torch.ones(batch_size, num_atoms, 1)

    graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)

    # Test the model
    hidden_size = 128
    depth = 3

    model = RCNN_WL_Last(hidden_size, depth)
    atom_hiddens, mol_fp = model(graph_inputs, batch_size)

    print(f"Input atom shape: {input_atom.shape}")
    print(f"Atom hiddens shape: {atom_hiddens.shape}")
    print(f"Molecular FP shape: {mol_fp.shape}")

    # Test functional version
    atom_hiddens_func, mol_fp_func = rcnn_wl_last(graph_inputs, batch_size, hidden_size, depth)
    print(f"\nFunctional version:")
    print(f"Atom hiddens shape: {atom_hiddens_func.shape}")
    print(f"Molecular FP shape: {mol_fp_func.shape}")
