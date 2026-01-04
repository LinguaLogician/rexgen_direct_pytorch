# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: mol_graph.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:00
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883
# [file name]: mol_graph.py
import torch
import rdkit.Chem as Chem
import numpy as np
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
             'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
             'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc',
             'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']

atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10


def onek_encoding_unk(x: str, allowable_set: List[str]) -> List[bool]:
    """One-hot encoding with unknown token."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom: Chem.Atom) -> np.ndarray:
    """Extract atom features."""
    return np.array(
        onek_encoding_unk(atom.GetSymbol(), elem_list) +
        onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6]) +
        onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()],
        dtype=np.float32
    )


def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Extract bond features."""
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ], dtype=np.float32)


def smiles2graph(smiles: str, idxfunc: Callable[[Chem.Atom], int] = lambda x: x.GetIdx()) -> Tuple[np.ndarray, ...]:
    """Convert SMILES to graph representation."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"Could not parse SMILES string: {smiles}")

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)

    fatoms = np.zeros((n_atoms, atom_fdim), dtype=np.float32)
    fbonds = np.zeros((n_bonds, bond_fdim), dtype=np.float32)
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    # Extract atom features
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx >= n_atoms:
            raise Exception(f"Atom index out of bounds in SMILES: {smiles}")
        fatoms[idx] = atom_features(atom)

    # Extract bond features and build adjacency
    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()

        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(f"Too many neighbors in SMILES: {smiles}")

        atom_nb[a1, num_nbs[a1]] = a2
        atom_nb[a2, num_nbs[a2]] = a1
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs


def pack2D(arr_list: List[np.ndarray]) -> np.ndarray:
    """Pad 2D arrays to same dimensions."""
    N = max(x.shape[0] for x in arr_list)
    M = max(x.shape[1] for x in arr_list)
    a = np.zeros((len(arr_list), N, M), dtype=arr_list[0].dtype)

    for i, arr in enumerate(arr_list):
        n, m = arr.shape
        a[i, :n, :m] = arr

    return a


def pack2D_withidx(arr_list: List[np.ndarray]) -> np.ndarray:
    """Pad 2D arrays with batch indices."""
    N = max(x.shape[0] for x in arr_list)
    M = max(x.shape[1] for x in arr_list)
    a = np.zeros((len(arr_list), N, M, 2), dtype=np.float32)

    for i, arr in enumerate(arr_list):
        n, m = arr.shape
        a[i, :n, :m, 0] = i
        a[i, :n, :m, 1] = arr

    return a


def pack1D(arr_list: List[np.ndarray]) -> np.ndarray:
    """Pad 1D arrays to same length."""
    N = max(x.shape[0] for x in arr_list)
    a = np.zeros((len(arr_list), N), dtype=arr_list[0].dtype)

    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i, :n] = arr

    return a


def get_mask(arr_list: List[np.ndarray]) -> np.ndarray:
    """Create mask for padded arrays."""
    N = max(x.shape[0] for x in arr_list)
    a = np.zeros((len(arr_list), N), dtype=np.float32)

    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i, :n] = 1.0

    return a


def smiles2graph_list(smiles_list: List[str],
                      idxfunc: Callable[[Chem.Atom], int] = lambda x: x.GetIdx(),
                      show_progress: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Convert list of SMILES to batched graph representation.

    Args:
        smiles_list: List of SMILES strings
        idxfunc: Function to get atom index
        show_progress: Whether to show progress bar

    Returns:
        Tuple of batched graph tensors
    """
    if show_progress:
        smiles_iter = tqdm(smiles_list, desc="Processing SMILES")
    else:
        smiles_iter = smiles_list

    res = []
    for smiles in smiles_iter:
        try:
            graph_data = smiles2graph(smiles, idxfunc)
            res.append(graph_data)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            raise

    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)

    return (pack2D(fatom_list),
            pack2D(fbond_list),
            pack2D_withidx(gatom_list),
            pack2D_withidx(gbond_list),
            pack1D(nb_list),
            get_mask(fatom_list))


if __name__ == "__main__":
    # Test the module
    np.set_printoptions(threshold=np.inf)
    a, b, c, d, e, f = smiles2graph_list(
        ["c1cccnc1", "c1nccc2n1ccc2"],
        show_progress=True
    )
    print(f"Atom features shape: {a.shape}")
    print(f"Bond features shape: {b.shape}")
    print(f"Atom graph shape: {c.shape}")
    print(f"Bond graph shape: {d.shape}")
    print(f"Num neighbors shape: {e.shape}")
    print(f"Mask shape: {f.shape}")
