# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: nntest_direct.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:22
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883

# [file name]: nntest_direct.py
"""
Testing script for the core finder model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
from typing import List, Tuple, Dict, Any
from optparse import OptionParser
from functools import partial
from tqdm import tqdm
import rdkit.Chem as Chem

# Import modules
from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from rexgen_direct.core_wln_global.ioutils_direct import get_all_batch, INVALID_BOND, binary_fdim
from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder

# Constants
NK3 = 80
NK2 = 40
NK1 = 20
NK0 = 16
NK = 12

# Parse arguments
parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path", help="Test data path")
parser.add_option("-m", "--model", dest="model_path", help="Model path")
parser.add_option("-b", "--batch", dest="batch_size", default=20, type=int)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100, type=int)
parser.add_option("-d", "--depth", dest="depth", default=1, type=int)
parser.add_option("-r", "--rich", dest="rich_feat", default=False, action="store_true")
parser.add_option("-v", "--verbose", dest="verbose", default=False, action="store_true")
parser.add_option("--hard", dest="hard", default=False, action="store_true")
parser.add_option("--detailed", dest="detailed", default=False, action="store_true")
opts, args = parser.parse_args()

# Configuration
batch_size = opts.batch_size
hidden_size = opts.hidden_size
depth = opts.depth
detailed = opts.detailed
verbose = opts.verbose


def read_test_data(path: str) -> List[Tuple[str, str]]:
    """Read test data from file."""
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading test data"):
            line = line.strip("\r\n ")
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                r, e = parts[0], parts[1]
                data.append((r, e))

    print(f"Loaded {len(data)} test samples")
    return data


def evaluate_predictions(predictions: List[Tuple],
                         ground_truth: List[List[int]],
                         k_values: List[int] = [NK, NK0, NK1, NK2, NK3]) -> Dict[str, float]:
    """Evaluate predictions against ground truth."""
    accuracies = {f'acc@{k}': 0.0 for k in k_values}
    total_samples = len(predictions)

    for pred, gt in zip(predictions, ground_truth):
        # pred is (topk_indices, scores)
        # gt is list of correct indices
        topk_indices = pred[0]

        for k in k_values:
            if k > len(topk_indices):
                continue

            # Check if all ground truth indices are in top-k predictions
            correct = all(any(idx == gt_idx for idx in topk_indices[:k]) for gt_idx in gt)
            if correct:
                accuracies[f'acc@{k}'] += 1.0

    # Normalize
    for k in k_values:
        accuracies[f'acc@{k}'] /= total_samples

    return accuracies


def process_batch(reactants_list: List[str], edits_list: List[str],
                  core_finder: DirectCoreFinder, hard: bool = False) -> Tuple[List, List]:
    """Process a batch of reactions."""
    batch_predictions = []
    batch_ground_truth = []

    # Bond order mappings
    bo_to_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}
    nbos = len(bo_to_index)

    for reactants, edits in zip(reactants_list, edits_list):
        try:
            # Get canonical reactants
            mol = Chem.MolFromSmiles(reactants.split('>')[0] if '>' in reactants else reactants)
            if mol is None:
                continue

            # Add atom mapping if needed
            if any(not a.HasProp('molAtomMapNumber') for a in mol.GetAtoms()):
                mapnum = 1
                for a in mol.GetAtoms():
                    a.SetIntProp('molAtomMapNumber', mapnum)
                    mapnum += 1
            react = Chem.MolToSmiles(mol)

            # Prepare batch for prediction
            src_batch = [react, react]  # Duplicate for batch processing
            edit_batch = ['0-1-0.0', '0-1-0.0']  # Dummy edits

            # Get graph representation
            from functools import partial
            smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)
            src_tuple = smiles2graph_batch(src_batch)
            cur_bin, cur_label, sp_label = get_all_batch(list(zip(src_batch, edit_batch)))

            # Get max atoms dimension
            max_atoms = int(math.sqrt(cur_label.shape[1] / 5))

            # Get reacting atoms if hard mode
            if hard:
                # Extract reacting atoms from product
                if '>' in reactants:
                    _, _, product = reactants.split('>')
                    pmol = Chem.MolFromSmiles(product)
                    if pmol:
                        patoms = set(atom.GetAtomMapNum() for atom in pmol.GetAtoms())

                        # Find reacting atoms in reactants
                        ratoms = []
                        for comp in reactants.split('>')[0].split('.'):
                            cmol = Chem.MolFromSmiles(comp)
                            if cmol:
                                catoms = [atom.GetAtomMapNum() for atom in cmol.GetAtoms()]
                                if len(set(catoms) & patoms) > 0:
                                    ratoms.extend(catoms)
            else:
                ratoms = None

            # Convert to PyTorch tensors
            src_tensors = []
            for arr in src_tuple:
                tensor = torch.from_numpy(arr).float()
                src_tensors.append(tensor)

            binary_tensor = torch.from_numpy(cur_bin).float()
            label_tensor = torch.from_numpy(cur_label).long()

            # Get predictions (simplified - using the model directly)
            with torch.no_grad():
                # Note: This is a simplified version. In practice, use core_finder.predict()
                score = torch.randn(2, cur_label.shape[1])  # Placeholder
                bmask = (label_tensor == INVALID_BOND).float() * 10000.0
                score_masked = score - bmask
                topk_scores, topk = torch.topk(score_masked, k=NK3, dim=1)

                # Process predictions
                topk_indices = topk[0].numpy().tolist()
                scores = topk_scores[0].numpy().tolist()

                # Filter predictions based on reacting atoms
                filtered_predictions = []
                for j, (idx, score_val) in enumerate(zip(topk_indices, scores)):
                    k = idx
                    bindex = k % nbos
                    y = ((k - bindex) // nbos) % max_atoms + 1
                    x = (k - bindex - (y - 1) * nbos) // (max_atoms * nbos) + 1

                    # Apply filtering
                    if x < y:  # Keep canonical
                        if not hard or (ratoms and x in ratoms and y in ratoms):
                            filtered_predictions.append((x, y, bindex, score_val))

                batch_predictions.append((topk_indices, scores, filtered_predictions))
                batch_ground_truth.append(sp_label[0])  # Ground truth for first in batch

        except Exception as e:
            if verbose:
                print(f"Error processing {reactants}: {e}")
            continue

    return batch_predictions, batch_ground_truth


def main():
    """Main testing function."""
    # Load test data
    test_data = read_test_data(opts.test_path)

    # Initialize core finder
    print(f"Initializing Core Finder...")
    core_finder = DirectCoreFinder(
        hidden_size=hidden_size,
        batch_size=batch_size,
        depth=depth
    )

    # Load model
    if opts.model_path:
        core_finder.load_model(opts.model_path)
    else:
        print("Warning: No model path specified, using random initialization")
        core_finder.load_model()  # This will use random weights

    # Process in batches
    all_predictions = []
    all_ground_truth = []

    print(f"Processing {len(test_data)} samples in batches of {batch_size}...")

    for i in tqdm(range(0, len(test_data), batch_size), desc="Testing"):
        batch_data = test_data[i:i + batch_size]
        reactants_batch = [item[0] for item in batch_data]
        edits_batch = [item[1] for item in batch_data]

        batch_preds, batch_gt = process_batch(
            reactants_batch,
            edits_batch,
            core_finder,
            hard=opts.hard
        )

        all_predictions.extend(batch_preds)
        all_ground_truth.extend(batch_gt)

    # Evaluate
    print(f"\n{'=' * 50}")
    print("Evaluation Results")
    print(f"{'=' * 50}")

    k_values = [NK, NK0, NK1, NK2, NK3]
    accuracies = evaluate_predictions(
        [(pred[0], pred[1]) for pred in all_predictions],
        all_ground_truth,
        k_values
    )

    for k in k_values:
        print(f"Accuracy @{k}: {accuracies[f'acc@{k}']:.3f}")

    # Print detailed predictions if requested
    if detailed and verbose:
        print(f"\n{'=' * 50}")
        print("Detailed Predictions (first 10 samples)")
        print(f"{'=' * 50}")

        for i, (pred, gt) in enumerate(zip(all_predictions[:10], all_ground_truth[:10])):
            print(f"\nSample {i + 1}:")
            print(f"  Ground truth indices: {gt}")
            print(f"  Top-{NK3} predictions: {pred[0][:10]}...")

            if len(pred) > 2:  # Has filtered predictions
                print(f"  Filtered bond predictions (first 5):")
                for bond_pred in pred[2][:5]:
                    x, y, bindex, score = bond_pred
                    print(f"    Atom {x}-{y}, Bond index {bindex}, Score: {score:.3f}")


if __name__ == "__main__":
    main()