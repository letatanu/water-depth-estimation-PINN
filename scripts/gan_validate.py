# scripts/gan_validate.py
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.gan_dataset import DeepFloodNPZDataset
from models.gan_models import UNetGenerator    

# --------------------------
# Metrics
# --------------------------

def mse_rmse_psnr(pred, target, max_val=1.0):
    """
    pred, target: [B,C,H,W] normalized in [0,1] (or Tanh mapped to [-1,1] then renormalized)
    """
    mse = F.mse_loss(pred, target, reduction="mean")
    rmse = torch.sqrt(mse + 1e-12)
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse + 1e-12)
    return mse.item(), rmse.item(), psnr.item()


# --------------------------
# Validation
# --------------------------

def validate_gan(
    checkpoint_path: str,
    npz_path: str,
    batch_size: int = 8,
    device: str = "cuda",
    max_val: float = 1.0,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    print("Loading dataset:", npz_path)
    val_ds = DeepFloodNPZDataset(npz_path, split="val", normalize=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---- Load generator ----
    print("Loading generator checkpoint:", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Your model:
    netG = UNetGenerator(in_channels=5, out_channels=3).to(device)

    #  "netG_state", "netD_state", "optG_state" ...  from gan_train.py 
    if "netG_state" in ckpt:
        netG.load_state_dict(ckpt["netG_state"])
    else:
        # if someone saved only the state_dict directly
        netG.load_state_dict(ckpt)

    netG.eval()

    # ---- Metrics accumulators ----
    mse_total, rmse_total, psnr_total = 0.0, 0.0, 0.0
    n_batches = 0

    # ------ Validation loop ------
    with torch.no_grad():
        for inp, target in val_loader:
            inp = inp.to(device)         # [B,5,H,W]
            target = target.to(device)   # [B,3,H,W]

            fake = netG(inp)             # output is Tanh() in [-1,1]


            mse, rmse, psnr = mse_rmse_psnr(fake, target, max_val=max_val)

            mse_total += mse
            rmse_total += rmse
            psnr_total += psnr
            n_batches += 1

    mse_avg = mse_total / n_batches
    rmse_avg = rmse_total / n_batches
    psnr_avg = psnr_total / n_batches

    print("\n===== GAN VALIDATION RESULTS =====")
    print(f"MSE  : {mse_avg:.6f}")
    print(f"RMSE : {rmse_avg:.6f}")
    print(f"PSNR : {psnr_avg:.2f} dB")
    print("=================================")

    return mse_avg, rmse_avg, psnr_avg


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="gan_checkpoints/epoch_200.pth")
    parser.add_argument("--npz_path", type=str, default="dataset/deepflood_anuga_dataset.npz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_val", type=float, default=1.0)
    args = parser.parse_args()

    validate_gan(
        checkpoint_path=args.checkpoint,
        npz_path=args.npz_path,
        batch_size=args.batch_size,
        device=args.device,
        max_val=args.max_val,
    )


if __name__ == "__main__":
    main()
