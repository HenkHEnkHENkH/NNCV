import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()

        # Encoder (Downsampling)
        self.inc = ResidualConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(256, 512))
        
        # Bottleneck + Dropout
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), 
            ResidualConv(512, 1024),
            nn.Dropout2d(p=0.2) 
        )

        # Decoder
        # Each ConvTranspose reduces channels by half, then we concat the skip connection
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.res_up1 = ResidualConv(512 + 512, 512) # 512 (up) + 512 (s4)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.res_up2 = ResidualConv(256 + 256, 256) # 256 (up) + 256 (s3)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.res_up3 = ResidualConv(128 + 128, 128) # 128 (up) + 128 (s2)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res_up4 = ResidualConv(64 + 64, 64)    # 64 (up) + 64 (s1)

        # Final Prediction Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        
        # Bottleneck
        b = self.down4(s4)

        # Decoder
        x = self.up1(b)
        x = torch.cat([x, s4], dim=1)
        x = self.res_up1(x)

        x = self.up2(x)
        x = torch.cat([x, s3], dim=1)
        x = self.res_up2(x)

        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.res_up3(x)

        x = self.up4(x)
        x = torch.cat([x, s1], dim=1)
        x = self.res_up4(x)

        return self.outc(x)

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block(x)
        return F.relu(x + residual)