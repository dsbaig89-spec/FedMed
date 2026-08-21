import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolution layers.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    Lightweight 3D U-Net for brain tumor segmentation.

    Input:
        [batch, 1, depth, height, width]

    Output:
        [batch, 1, depth, height, width]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = DoubleConv3D(
            in_channels,
            16
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.enc2 = DoubleConv3D(
            16,
            32
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = DoubleConv3D(
            32,
            64
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv3D(
            64,
            32
        )

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv3D(
            32,
            16
        )

        # -------------------------
        # Final segmentation layer
        # -------------------------

        self.final = nn.Conv3d(
            16,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool2(e2)
        )

        # Decoder
        d2 = self.up2(b)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # Output
        output = self.final(d1)

        return output


if __name__ == "__main__":

    print("Testing FedMed 3D U-Net...")

    model = UNet3D()

    # Dummy 3D MRI volume
    x = torch.randn(
        1,
        1,
        32,
        32,
        32
    )

    with torch.no_grad():
        output = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("\n3D U-Net test successful!")