# gan_train.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.gan_dataset import DeepFloodNPZDataset
from models.gan_models import UNetGenerator, PatchDiscriminator


def train_gan(
    npz_path: str = "dataset/deepflood_anuga_dataset_clean.npz",
    batch_size: int = 8,
    num_epochs: int = 50,
    lr: float = 2e-4,
    lambda_L1: float = 100.0,
    device: str = None,
    out_dir: str = "gan_checkpoints",
):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets / loaders
    train_ds = DeepFloodNPZDataset(npz_path, split="train", normalize=True)
    val_ds = DeepFloodNPZDataset(
        npz_path,
        split="val",
        normalize=True,
        channel_means=(train_ds.mean_X, train_ds.mean_Y),
        channel_stds=(train_ds.std_X, train_ds.std_Y),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Models
    netG = UNetGenerator(in_channels=5, out_channels=3).to(device)
    netD = PatchDiscriminator(in_channels=5, out_channels=3).to(device)

    # Losses
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        netG.train()
        netD.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        
        for i, (inp, target) in enumerate(train_loader):
            inp = inp.to(device)       # [B, 5, H, W]
            target = target.to(device) # [B, 3, H, W]

            # ----------------------
            #  Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()

            # Real pair
            pred_real = netD(inp, target)          # [B,1,h,w]
            real_label = torch.ones_like(pred_real)  # match shape!
            loss_D_real = criterion_GAN(pred_real, real_label)

            # Fake pair
            with torch.no_grad():
                fake_detached = netG(inp)
            pred_fake = netD(inp, fake_detached)
            fake_label = torch.zeros_like(pred_fake)
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            # ----------------------
            #  Train Generator
            # ----------------------
            optimizer_G.zero_grad()

            fake = netG(inp)
            pred_fake_for_G = netD(inp, fake)
            real_label_for_G = torch.ones_like(pred_fake_for_G)

            loss_G_GAN = criterion_GAN(pred_fake_for_G, real_label_for_G)
            loss_G_L1 = criterion_L1(fake, target) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()
            optimizer_G.step()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()

        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f}")

        # TODO: add validation, early stopping, etc.

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "netG_state": netG.state_dict(),
                "netD_state": netD.state_dict(),
                "optG_state": optimizer_G.state_dict(),
                "optD_state": optimizer_D.state_dict(),
            },
            out_dir / f"epoch_{epoch+1:03d}.pth",
        )


if __name__ == "__main__":
    train_gan()
