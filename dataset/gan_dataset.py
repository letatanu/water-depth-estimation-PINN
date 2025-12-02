# gan_dataset.py
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DeepFloodNPZDataset(Dataset):
    """
    Dataset for DeepFlood-style NPZ:
      X: [N, 5, H, W]
      Y: [N, 3, H, W]

    Returns (input, target) as float32 tensors:
      input:  [5, H, W]
      target: [3, H, W]
    """

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        normalize: bool = True,
        channel_means: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        channel_stds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            npz_path: path to deepflood_anuga_dataset.npz
            split: 'train' | 'val' | 'test'
            train_ratio: fraction of samples for training
            val_ratio: fraction for validation
            normalize: if True, apply per-channel normalization
            channel_means, channel_stds:
                optional (mean_X, mean_Y), (std_X, std_Y); each shape [C]
                if None and normalize=True, they are computed from the training subset.
        """
        super().__init__()
        self.npz_path = Path(npz_path)
        self.split = split
        self.normalize = normalize

        data = np.load(self.npz_path)
        X = data["X"]  # [N, 5, H, W]
        Y = data["Y"]  # [N, 3, H, W]
        N = X.shape[0]

        # simple random split indices
        rng = np.random.RandomState(1234)
        idx = np.arange(N)
        rng.shuffle(idx)

        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
        n_test = N - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        if split == "train":
            self.X = X[train_idx]
            self.Y = Y[train_idx]
        elif split == "val":
            self.X = X[val_idx]
            self.Y = Y[val_idx]
        elif split == "test":
            self.X = X[test_idx]
            self.Y = Y[test_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Compute or store normalization stats
        if normalize:
            if channel_means is None or channel_stds is None:
                # compute from *training* subset only
                if split != "train":
                    # load train subset to compute stats
                    train_data = np.load(self.npz_path)
                    X_train = train_data["X"][train_idx]
                    Y_train = train_data["Y"][train_idx]

                    mean_X = X_train.mean(axis=(0, 2, 3))  # [5]
                    std_X = X_train.std(axis=(0, 2, 3)) + 1e-6
                    mean_Y = Y_train.mean(axis=(0, 2, 3))  # [3]
                    std_Y = Y_train.std(axis=(0, 2, 3)) + 1e-6
                else:
                    mean_X = self.X.mean(axis=(0, 2, 3))
                    std_X = self.X.std(axis=(0, 2, 3)) + 1e-6
                    mean_Y = self.Y.mean(axis=(0, 2, 3))
                    std_Y = self.Y.std(axis=(0, 2, 3)) + 1e-6
            else:
                mean_X, mean_Y = channel_means
                std_X, std_Y = channel_stds

            self.mean_X = mean_X.astype(np.float32)
            self.std_X = std_X.astype(np.float32)
            self.mean_Y = mean_Y.astype(np.float32)
            self.std_Y = std_Y.astype(np.float32)
        else:
            self.mean_X = None
            self.std_X = None
            self.mean_Y = None
            self.std_Y = None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx].astype(np.float32)  # [5, H, W]
        y = self.Y[idx].astype(np.float32)  # [3, H, W]

        if self.normalize:
            x = (x - self.mean_X[:, None, None]) / self.std_X[:, None, None]
            y = (y - self.mean_Y[:, None, None]) / self.std_Y[:, None, None]

        x_tensor = torch.from_numpy(x)  # [5, H, W]
        y_tensor = torch.from_numpy(y)  # [3, H, W]
        return x_tensor, y_tensor
