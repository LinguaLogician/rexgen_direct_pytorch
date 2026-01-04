# -*- coding: utf-8 -*-
# @project: rexgen_direct_pytorch
# @filename: nntrain_direct.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2026/1/3 10:14
# https://chat.deepseek.com/a/chat/s/510b5a4b-d7b4-45e0-933b-692a1f214883
# [file name]: nntrain_direct.py
"""
Training script for the core finder model.
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import sys
import random
import os
from collections import Counter
from optparse import OptionParser
from functools import partial
from tqdm import tqdm
# 在 nntrain_direct.py 开头添加
from rexgen_direct.core_wln_global.rdkit_config import configure_rdkit_logging
configure_rdkit_logging('ERROR')  # 只显示错误信息
# Import modules
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rexgen_direct.core_wln_global.nn import linearND, linear
from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from rexgen_direct.core_wln_global.models import rcnn_wl_last
from rexgen_direct.core_wln_global.ioutils_direct import get_all_batch, INVALID_BOND, binary_fdim

# Constants
NK = 20
NK0 = 10

# Create function for batch processing
smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)


class ReactionDataset(Dataset):
    """Dataset for reaction data."""

    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, 'r') as f:
            for line in tqdm(f, desc="Loading data"):
                line = line.strip("\r\n ")
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    r, e = parts[0], parts[1]
                    self.data.append((r, e))

        print(f"Loaded {len(self.data)} reactions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DirectCoreFinderTrainer:
    """Trainer for the direct core finder model."""

    def __init__(self, hidden_size: int, depth: int, batch_size: int,
                 atom_fdim: int = None, bond_fdim: int = None):
        self.hidden_size = hidden_size
        self.depth = depth
        self.batch_size = batch_size

        # 如果未提供维度，使用默认值
        if atom_fdim is None:
            from rexgen_direct.core_wln_global.mol_graph import atom_fdim as default_adim
            atom_fdim = default_adim
        if bond_fdim is None:
            from rexgen_direct.core_wln_global.mol_graph import bond_fdim as default_bdim
            bond_fdim = default_bdim

        # Initialize model
        from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinderModel
        self.model = DirectCoreFinderModel(
            hidden_size=hidden_size,
            depth=depth,
            atom_fdim=atom_fdim,
            bond_fdim=bond_fdim
        )

        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # 这将把整个模型移到设备上

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model size: {total_params / 1000:.1f}K parameters")
        print(f"Trainable parameters: {trainable_params / 1000:.1f}K")
        print(f"Input dimensions - Atom: {atom_fdim}, Bond: {bond_fdim}")
        print(f"Using device: {self.device}")

    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer,
                    epoch: int, max_norm: float = 5.0) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc_nk = 0.0
        total_acc_nk0 = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch_data in pbar:
            # batch_data 现在已经是批次的列表
            if not batch_data:
                continue

            # 提取批次数据
            src_batch = []
            edit_batch = []
            for item in batch_data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    r, e = item[0], item[1]
                    src_batch.append(r)
                    edit_batch.append(e)

            if len(src_batch) == 0:
                continue

            # 确保批次大小正确
            if len(src_batch) != self.batch_size:
                # 如果是最后一个不完整的批次，跳过
                continue

            # 继续原来的处理逻辑...
            # Extract reactants from reaction string
            processed_batch = []
            for r in src_batch:
                # Handle reaction format: reactant>agent>product
                if '>' in r:
                    react = r.split('>')[0]
                else:
                    react = r
                processed_batch.append(react)

            # Prepare graph inputs
            src_tuple = smiles2graph_batch(list(processed_batch))
            cur_bin, cur_label, sp_label = get_all_batch(list(zip(processed_batch, edit_batch)))

            # Convert to tensors
            src_tensors = []
            for arr in src_tuple:
                tensor = torch.from_numpy(arr).float().to(self.device)
                src_tensors.append(tensor)

            binary_tensor = torch.from_numpy(cur_bin).float().to(self.device)
            label_tensor = torch.from_numpy(cur_label).long().to(self.device)

            # Forward pass
            outputs = self.model(tuple(src_tensors), binary_tensor, label_tensor)
            score = outputs['score']

            # Calculate loss
            flat_score = score.reshape(-1)
            flat_label = label_tensor.reshape(-1)
            bond_mask = (flat_label != INVALID_BOND).float()
            flat_label_clamped = torch.clamp(flat_label, min=0)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(
                flat_score, flat_label_clamped.float(),
                reduction='none'
            )
            loss = torch.sum(loss * bond_mask) / batch_size

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                # Get top-k predictions
                bmask = (label_tensor == INVALID_BOND).float() * 10000.0
                score_masked = score - bmask
                topk_scores, topk = torch.topk(score_masked, k=NK, dim=1)

                batch_acc_nk = 0.0
                batch_acc_nk0 = 0.0

                for i in range(batch_size):
                    # Count correct predictions
                    pre_nk = sum(1 for j in range(NK) if topk[i, j].item() in sp_label[i])
                    pre_nk0 = sum(1 for j in range(NK0) if topk[i, j].item() in sp_label[i])

                    if len(sp_label[i]) == pre_nk:
                        batch_acc_nk += 1
                    if len(sp_label[i]) == pre_nk0:
                        batch_acc_nk0 += 1

                total_acc_nk += batch_acc_nk
                total_acc_nk0 += batch_acc_nk0
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc@NK': f'{batch_acc_nk / batch_size:.3f}',
                'acc@NK0': f'{batch_acc_nk0 / batch_size:.3f}'
            })

        return {
            'loss': total_loss / total_samples,
            'acc_nk': total_acc_nk / total_samples,
            'acc_nk0': total_acc_nk0 / total_samples
        }

    def save_model(self, path: str, epoch: int):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'hidden_size': self.hidden_size,
            'depth': self.depth,
            'batch_size': self.batch_size
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            return checkpoint.get('epoch', 0)
        else:
            print(f"Checkpoint not found at {path}")
            return 0


class BucketDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # idx 可能是整数，也可能是列表（当使用 BatchSampler 时）
        if isinstance(idx, (list, tuple)):
            # 如果是列表，返回多个样本
            return [self.dataset[i] for i in idx]
        else:
            # 如果是整数，返回单个样本
            return self.dataset[self.indices[idx]]


# 修改 BucketSampler 类
class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, buckets, batch_size, shuffle=True):
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = 0

        # 计算总长度
        for bucket in self.buckets:
            self.length += len(bucket) // batch_size

    def __iter__(self):
        # 为每个桶创建索引
        all_batches = []

        for bucket_indices in self.buckets:
            if len(bucket_indices) == 0:
                continue

            if self.shuffle:
                random.shuffle(bucket_indices)

            # 创建批次
            batches = []
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

            all_batches.extend(batches)

        # 如果需要，打乱批次顺序
        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.length


# 修改 DataLoader 创建方式
def create_data_loaders(train_path: str, batch_size: int, shuffle: bool = True):
    """Create data loaders with bucketing by molecule size."""
    # Load all data first
    dataset = ReactionDataset(train_path)

    # Create buckets based on molecule size
    buckets = [[] for _ in range(10)]  # 10 buckets
    bucket_sizes = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]

    for idx in range(len(dataset)):
        r, _ = dataset[idx]
        # Count heavy atoms
        if '>' in r:
            react_part = r.split('>')[0]
        else:
            react_part = r

        mol = Chem.MolFromSmiles(react_part)
        if mol:
            num_atoms = mol.GetNumAtoms()
            for i, size in enumerate(bucket_sizes):
                if num_atoms <= size:
                    buckets[i].append(idx)
                    break

    # 创建一个包装器数据集来处理批次索引
    class BatchWrapperDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            if isinstance(idx, (list, tuple)):
                return [self.dataset[i] for i in idx]
            return self.dataset[idx]

    # 创建包装器数据集
    wrapped_dataset = BatchWrapperDataset(dataset)

    # 创建批次采样器
    batch_sampler = BucketSampler(buckets, batch_size, shuffle)

    # 创建 dataloader
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=None,  # 设置为 None，因为 BatchSampler 会处理批次
        sampler=batch_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda x: x  # 直接返回批次，不进行额外处理
    )

    return dataloader


def main():
    """Main training function."""
    # Create save directory
    os.makedirs(opts.save_path, exist_ok=True)

    # Initialize trainer
    trainer = DirectCoreFinderTrainer(
        hidden_size=hidden_size,
        depth=depth,
        batch_size=batch_size
    )

    # Create data loader
    print("Creating data loader...")
    train_loader = create_data_loaders(opts.train_path, batch_size, shuffle=True)

    # Setup optimizer
    optimizer = optim.Adam(trainer.model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)

    # Training loop
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'=' * 50}")

        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, optimizer, epoch, max_norm)

        # Print metrics
        print(f"Epoch {epoch} - Loss: {metrics['loss']:.4f}, "
              f"Acc@NK: {metrics['acc_nk']:.3f}, "
              f"Acc@NK0: {metrics['acc_nk0']:.3f}")

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                opts.save_path,
                f"model_epoch_{epoch}.pt"
            )
            trainer.save_model(checkpoint_path, epoch)

    # Save final model
    final_path = os.path.join(opts.save_path, "model_final.pt")
    trainer.save_model(final_path, epochs)

    print(f"\nTraining completed!")
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":

    # --train ../data/train.txt.proc --hidden 300 --depth 3 --save_dir model-300-3-direct
    # Parse arguments
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path",
                      default="../data/train.txt.proc",
                      help="Training data path")
    parser.add_option("-m", "--save_dir", dest="save_path",
                      default="model-300-3-direct",
                      help="Model save directory")
    parser.add_option("-b", "--batch", dest="batch_size", default=20, type=int)
    parser.add_option("-w", "--hidden", dest="hidden_size", default=300, type=int)
    parser.add_option("-d", "--depth", dest="depth", default=3, type=int)
    parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0, type=float)
    parser.add_option("-r", "--rich", dest="rich_feat", default=False, action="store_true")
    parser.add_option("-e", "--epochs", dest="epochs", default=100, type=int)
    parser.add_option("--lr", dest="learning_rate", default=0.001, type=float)
    opts, args = parser.parse_args()

    # Configuration
    batch_size = opts.batch_size
    hidden_size = opts.hidden_size
    depth = opts.depth
    max_norm = opts.max_norm
    epochs = opts.epochs
    learning_rate = opts.learning_rate

    # Select feature type
    if opts.rich_feat:
        print("Using rich atom features")
        from rexgen_direct.core_wln_global.mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, \
            smiles2graph_list as _s2g
    else:
        from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, \
            smiles2graph_list as _s2g

    main()
