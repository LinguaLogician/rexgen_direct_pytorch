# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: ioutils_direct.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:12
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: ioutils_direct.py
import rdkit.Chem as Chem
import numpy as np
from typing import List, Tuple, Dict, Any
from rexgen_direct.core_wln_global.mol_graph import bond_fdim, bond_features
from tqdm import tqdm

BOND_TYPE = ["NOBOND",
             Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
N_BOND_CLASS = len(BOND_TYPE)
binary_fdim = 4 + bond_fdim
INVALID_BOND = -1


def get_bin_feature(r: str, max_natoms: int) -> np.ndarray:
    """
    Generate descriptions of atom-atom relationships for global attention mechanism.

    Args:
        r: SMILES string
        max_natoms: Maximum number of atoms for padding

    Returns:
        Binary features tensor of shape (max_natoms, max_natoms, binary_fdim)
    """
    # Determine component membership for each atom
    comp = {}
    for i, s in enumerate(r.split('.')):
        mol = Chem.MolFromSmiles(s)
        for atom in mol.GetAtoms():
            comp[atom.GetIntProp('molAtomMapNumber') - 1] = i

    n_comp = len(r.split('.'))
    rmol = Chem.MolFromSmiles(r)
    n_atoms = rmol.GetNumAtoms()

    # Build bond map
    bond_map = {}
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        bond_map[(a1, a2)] = bond
        bond_map[(a2, a1)] = bond

    # Build features matrix
    features = []
    for i in range(max_natoms):
        for j in range(max_natoms):
            f = np.zeros((binary_fdim,), dtype=np.float32)

            if i >= n_atoms or j >= n_atoms or i == j:
                features.append(f)
                continue

            if (i, j) in bond_map:
                bond = bond_map[(i, j)]
                f[1:1 + bond_fdim] = bond_features(bond)
            else:
                f[0] = 1.0  # NOBOND feature

            # Component relationship features
            f[-4] = 1.0 if comp.get(i, -1) != comp.get(j, -1) else 0.0  # Different molecules
            f[-3] = 1.0 if comp.get(i, -1) == comp.get(j, -1) else 0.0  # Same molecule
            f[-2] = 1.0 if n_comp == 1 else 0.0  # Single molecule system
            f[-1] = 1.0 if n_comp > 1 else 0.0  # Multi-molecule system

            features.append(f)

    return np.vstack(features).reshape((max_natoms, max_natoms, binary_fdim))


bo_to_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}
nbos = len(bo_to_index)


def get_bond_label(r: str, edits: str, max_natoms: int) -> Tuple[np.ndarray, List[int]]:
    """
    Generate bond change labels for training.

    Args:
        r: Reactant SMILES
        edits: Bond edit string (format: "x-y-bo;...")
        max_natoms: Maximum number of atoms for padding

    Returns:
        Tuple of (labels array, sparse labels list)
    """
    rmol = Chem.MolFromSmiles(r)
    n_atoms = rmol.GetNumAtoms()

    # Initialize reaction map
    rmap = np.zeros((max_natoms, max_natoms, nbos), dtype=np.float32)

    # Parse edit string
    for edit in edits.split(';'):
        if not edit:
            continue
        a1_str, a2_str, bo_str = edit.split('-')
        x = min(int(a1_str) - 1, int(a2_str) - 1)
        y = max(int(a1_str) - 1, int(a2_str) - 1)
        z = bo_to_index[float(bo_str)]
        rmap[x, y, z] = 1.0
        rmap[y, x, z] = 1.0

    # Flatten labels
    labels = []
    sp_labels = []

    for i in range(max_natoms):
        for j in range(max_natoms):
            for k in range(nbos):
                if i == j or i >= n_atoms or j >= n_atoms:
                    labels.append(INVALID_BOND)  # Mask
                else:
                    label_val = int(rmap[i, j, k])
                    labels.append(label_val)
                    if label_val == 1:
                        # Flatten index calculation
                        sp_labels.append(i * max_natoms * nbos + j * nbos + k)

    return np.array(labels, dtype=np.int32), sp_labels


def get_all_batch(re_list: List[Tuple[str, str]],
                  show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """
    Process batch of reactant-edit pairs.

    Args:
        re_list: List of (reactant_smiles, edit_string) tuples
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (binary_features, labels, sparse_labels)
    """
    mol_list = []
    max_natoms = 0

    if show_progress:
        re_iter = tqdm(re_list, desc="Processing batch")
    else:
        re_iter = re_list

    # First pass: find max atoms
    for r, e in re_iter:
        rmol = Chem.MolFromSmiles(r)
        mol_list.append((r, e))
        max_natoms = max(max_natoms, rmol.GetNumAtoms())

    # Second pass: extract features and labels
    labels = []
    features = []
    sp_labels = []

    for r, e in mol_list:
        l, sl = get_bond_label(r, e, max_natoms)
        features.append(get_bin_feature(r, max_natoms))
        labels.append(l)
        sp_labels.append(sl)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), sp_labels


def get_feature_batch(r_list: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Extract binary features for a batch of reactants.

    Args:
        r_list: List of reactant SMILES
        show_progress: Whether to show progress bar

    Returns:
        Binary features tensor
    """
    max_natoms = 0

    if show_progress:
        r_iter = tqdm(r_list, desc="Extracting features")
    else:
        r_iter = r_list

    # Find max atoms
    for r in r_iter:
        rmol = Chem.MolFromSmiles(r)
        max_natoms = max(max_natoms, rmol.GetNumAtoms())

    # Extract features
    features = []
    for r in r_list:
        features.append(get_bin_feature(r, max_natoms))

    return np.array(features, dtype=np.float32)


# Test function
if __name__ == "__main__":
    # Test the functions
    test_reactant = "[CH3:1][CH2:2][OH:3]"
    test_edit = "1-2-1.0;2-3-0.0"

    # Add atom mapping if not present
    mol = Chem.MolFromSmiles(test_reactant)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp('molAtomMapNumber', i + 1)
    test_reactant = Chem.MolToSmiles(mol)

    print(f"Test reactant: {test_reactant}")
    print(f"Test edit: {test_edit}")

    # Test get_bin_feature
    max_atoms = mol.GetNumAtoms()
    bin_feat = get_bin_feature(test_reactant, max_atoms)
    print(f"Binary features shape: {bin_feat.shape}")

    # Test get_bond_label
    labels, sp_labels = get_bond_label(test_reactant, test_edit, max_atoms)
    print(f"Labels shape: {labels.shape}")
    print(f"Sparse labels: {sp_labels}")

    # Test get_all_batch
    batch = [(test_reactant, test_edit)]
    features_batch, labels_batch, sp_labels_batch = get_all_batch(batch, show_progress=True)
    print(f"Batch features shape: {features_batch.shape}")
    print(f"Batch labels shape: {labels_batch.shape}")
