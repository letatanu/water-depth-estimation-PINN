import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.gan_dataset import DeepFloodNPZDataset
from models.gan_models import UNetGenerator


def denormalize(tensor, mean, std):
    """
    tensor: [C, H, W] or [B, C, H, W] (torch)
    mean, std: [C] (numpy)
    """
    if tensor.dim() == 3:
        C, H, W = tensor.shape
        mean_t = torch.from_numpy(mean).view(C, 1, 1).to(tensor.device)
        std_t = torch.from_numpy(std).view(C, 1, 1).to(tensor.device)
    else:
        B, C, H, W = tensor.shape
        mean_t = torch.from_numpy(mean).view(1, C, 1, 1).to(tensor.device)
        std_t = torch.from_numpy(std).view(1, C, 1, 1).to(tensor.device)

    return tensor * std_t + mean_t


def visualize_samples(
    npz_path="dataset/deepflood_anuga_dataset.npz",
    ckpt_path="gan_checkpoints/epoch_050.pth",
    split="val",
    num_samples=5,
    out_dir="gan_vis",
    device=None,
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset (normalized)
    ds = DeepFloodNPZDataset(npz_path, split=split, normalize=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    print(f"Loaded dataset from {npz_path} split={split}, N={len(ds)}")

    # 2. Load generator
    netG = UNetGenerator(in_channels=5, out_channels=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(ckpt["netG_state"])
    netG.to(device)
    netG.eval()
    print(f"Loaded generator weights from {ckpt_path}")

    # 3. Get normalization stats for Y (targets)
    mean_Y = ds.mean_Y  # [3]
    std_Y = ds.std_Y    # [3]

    # Channel names for plotting
    ch_names = ["depth", "qx", "qy"]

    # 4. Loop over a few samples
    with torch.no_grad():
        for i, (inp, target) in enumerate(loader):
            if i >= num_samples:
                break

            inp = inp.to(device)         # [1, 5, H, W]
            target = target.to(device)   # [1, 3, H, W]

            # Forward pass through generator
            pred = netG(inp)             # [1, 3, H, W]

            # Denormalize target & prediction to physical scale
            target_den = denormalize(target, mean_Y, std_Y)  # [1,3,H,W]
            pred_den = denormalize(pred, mean_Y, std_Y)      # [1,3,H,W]

            # Remove batch dimension
            target_den = target_den[0].cpu().numpy()  # [3,H,W]
            pred_den = pred_den[0].cpu().numpy()      # [3,H,W]

            H, W = target_den.shape[1], target_den.shape[2]

            # 5. Plot GT vs Pred for all 3 channels
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            fig.suptitle(f"Sample {i} – Ground Truth vs GAN Output", fontsize=14)

            for c in range(3):
                gt = target_den[c]
                pr = pred_den[c]

                # Use shared color scale per channel
                vmin = min(gt.min(), pr.min())
                vmax = max(gt.max(), pr.max())

                # Top row: Ground Truth
                ax_gt = axes[0, c]
                im_gt = ax_gt.imshow(gt, origin="lower", vmin=vmin, vmax=vmax)
                ax_gt.set_title(f"GT {ch_names[c]}")
                ax_gt.set_xticks([]); ax_gt.set_yticks([])
                fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

                # Bottom row: Prediction
                ax_pr = axes[1, c]
                im_pr = ax_pr.imshow(pr, origin="lower", vmin=vmin, vmax=vmax)
                ax_pr.set_title(f"Pred {ch_names[c]}")
                ax_pr.set_xticks([]); ax_pr.set_yticks([])
                fig.colorbar(im_pr, ax=ax_pr, fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_path = out_dir / f"vis_sample_{i:03d}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"Saved visualization for sample {i} -> {out_path}")


if __name__ == "__main__":
    # Adjust paths if needed
    visualize_samples(
        npz_path="dataset/deepflood_anuga_dataset_clean.npz",
        ckpt_path="gan_checkpoints/epoch_050.pth",  
        split="val",
        num_samples=5,
        out_dir="gan_vis",
    )
