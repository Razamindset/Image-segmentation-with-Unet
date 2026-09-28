import torch
import torch.nn as nn
from encoder import EncoderBlock
from decoder import DecoderBlock
from double_conv import DoubleConv

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder Path reducing spacal dimension and increasing features
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottle neck
        self.bottleneck = DoubleConv(512, 1024)

        # Upscaling and recovering dims
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # go through the encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Final layer 
        out = self.final_conv(x)
        return out


if __name__ == "__main__":
    # Create a dummy RGB image batch: [Batch=1, Channels=3, Height=256, Width=256]
    x = torch.randn(1, 3, 256, 256)

    # Instantiate model for binary segmentation (e.g., foreground vs. background)
    model = UNet(in_channels=3, out_channels=1)

    # Run forward pass
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
