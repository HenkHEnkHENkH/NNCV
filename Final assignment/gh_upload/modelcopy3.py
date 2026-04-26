import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()

        # --- Encoder ---
        self.inc = ResidualConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualConv(256, 512))
        
        # --- Bottleneck + Dropout ---
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), 
            ResidualConv(512, 1024),
            nn.Dropout2d(p=0.2) 
        )

        # --- Decoder + Attention Gates ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256) # g from up1, l from s4
        self.res_up1 = ResidualConv(512 + 512, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128) # g from up2, l from s3
        self.res_up2 = ResidualConv(256 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)  # g from up3, l from s2
        self.res_up3 = ResidualConv(128 + 128, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)    # g from up4, l from s1
        self.res_up4 = ResidualConv(64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b  = self.down4(s4)

        # Decoder with Attention Filtering
        x = self.up1(b)
        s4_att = self.att1(g=x, x=s4) # Filter s4 using gate x
        x = torch.cat([x, s4_att], dim=1)
        x = self.res_up1(x)

        x = self.up2(x)
        s3_att = self.att2(g=x, x=s3)
        x = torch.cat([x, s3_att], dim=1)
        x = self.res_up2(x)

        x = self.up3(x)
        s2_att = self.att3(g=x, x=s2)
        x = torch.cat([x, s2_att], dim=1)
        x = self.res_up3(x)

        x = self.up4(x)
        s1_att = self.att4(g=x, x=s1)
        x = torch.cat([x, s1_att], dim=1)
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
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # W_g: Gating signal (from deeper layer)
        # W_l: Skip connection (from encoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # The 'psi' is a 0 to 1 mask that multiplies the skip connection
        return x * psi
    
