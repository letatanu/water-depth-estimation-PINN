# gan_models.py
import torch
import torch.nn as nn


# -----------------------------
# Basic building blocks
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, output_padding=0):
        """
        Generic upsampling block:
          ConvTranspose2d(in_ch -> out_ch, k=4, s=2, p=1, output_padding=output_padding)
        """
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net generator mapping 5-channel input -> 3-channel output, adapted for 100x100.

    Encoder sizes (H x W):
      e1: 100 -> 50   (ngf)
      e2: 50  -> 25   (2ngf)
      e3: 25  -> 12   (4ngf)
      e4: 12  -> 6    (8ngf)
      e5: 6   -> 3    (8ngf)

    Bottleneck: keeps 3x3.

    Decoder sizes:
      d1: 3  -> 6   (skip with e4: 6x6)  -> 16ngf
      d2: 6  -> 12  (skip with e3: 12x12)-> 12ngf
      d3: 12 -> 25  (skip with e2: 25x25)-> 6ngf   <-- uses output_padding=1
      d4: 25 -> 50  (skip with e1: 50x50)-> 3ngf
      d5: 50 -> 100 (no skip)            -> ngf

    Final conv keeps 100x100 and maps ngf -> out_channels.
    """

    def __init__(self, in_channels=5, out_channels=3, ngf=64):
        super().__init__()

        # ---------- Encoder ----------
        # 100 -> 50
        self.enc1 = ConvBlock(in_channels, ngf, norm=False)   # [B, 1*ngf, 50,50]
        # 50 -> 25
        self.enc2 = ConvBlock(ngf, ngf * 2)                   # [B, 2*ngf, 25,25]
        # 25 -> 12
        self.enc3 = ConvBlock(ngf * 2, ngf * 4)               # [B, 4*ngf, 12,12]
        # 12 -> 6
        self.enc4 = ConvBlock(ngf * 4, ngf * 8)               # [B, 8*ngf, 6,6]
        # 6 -> 3
        self.enc5 = ConvBlock(ngf * 8, ngf * 8)               # [B, 8*ngf, 3,3]

        # ---------- Bottleneck (3x3 -> 3x3) ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )

        # ---------- Decoder ----------
        # d1: 3 -> 6, then concat with e4 (6x6, 8ngf) => 16ngf
        self.dec1 = UpConvBlock(ngf * 8, ngf * 8, dropout=True, output_padding=0)

        # d2: 6 -> 12, then concat with e3 (12x12, 4ngf) => 12ngf
        self.dec2 = UpConvBlock(ngf * 16, ngf * 8, dropout=True, output_padding=0)

        # d3: 12 -> 25 (needs output_padding=1), then concat with e2 (25x25, 2ngf) => 6ngf
        self.dec3 = UpConvBlock(ngf * 12, ngf * 4, dropout=False, output_padding=1)

        # d4: 25 -> 50, then concat with e1 (50x50, 1ngf) => 3ngf
        self.dec4 = UpConvBlock(ngf * 6, ngf * 2, dropout=False, output_padding=0)

        # d5: 50 -> 100, no skip (3ngf -> ngf)
        self.dec5 = UpConvBlock(ngf * 3, ngf, dropout=False, output_padding=0)

        # final conv: keep 100x100, map ngf -> out_channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)  # [B, ngf,   50,50]
        e2 = self.enc2(e1) # [B, 2ngf,  25,25]
        e3 = self.enc3(e2) # [B, 4ngf,  12,12]
        e4 = self.enc4(e3) # [B, 8ngf,  6, 6]
        e5 = self.enc5(e4) # [B, 8ngf,  3, 3]

        b = self.bottleneck(e5)  # [B, 8ngf, 3,3]

        # ----- Decoder + skips -----
        d1 = self.dec1(b)           # [B, 8ngf, 6,6]
        d1 = torch.cat([d1, e4], 1) # [B, 16ngf,6,6]

        d2 = self.dec2(d1)          # [B, 8ngf, 12,12]
        d2 = torch.cat([d2, e3], 1) # [B, 12ngf,12,12]

        d3 = self.dec3(d2)          # [B, 4ngf, 25,25]   (because output_padding=1)
        d3 = torch.cat([d3, e2], 1) # [B, 6ngf, 25,25]

        d4 = self.dec4(d3)          # [B, 2ngf, 50,50]
        d4 = torch.cat([d4, e1], 1) # [B, 3ngf, 50,50]

        d5 = self.dec5(d4)          # [B, ngf,  100,100]

        out = self.final(d5)        # [B, out_channels, 100,100]
        return out

# -----------------------------
# PatchGAN Discriminator
# -----------------------------

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that sees concatenated (input, target) or (input, generated).
    Channels in = in_channels + out_channels = 5 + 3 = 8.
    """

    def __init__(self, in_channels=5, out_channels=3, ndf=64):
        super().__init__()
        C = in_channels + out_channels

        self.model = nn.Sequential(
            # no norm in first layer
            nn.Conv2d(C, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            # output: [B, 1, H/2^n, W/2^n], patch-wise real/fake
        )

    def forward(self, inp, out):
        # inp: [B, 5, H, W]
        # out: [B, 3, H, W]
        x = torch.cat([inp, out], dim=1)
        return self.model(x)
