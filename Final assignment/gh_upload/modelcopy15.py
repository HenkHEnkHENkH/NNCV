import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()

        # ===== ResNet50 Encoder =====
        resnet = models.resnext50_32x4d(weights='DEFAULT')

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )
        self.pool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aspp = ASPP(2048, 1024, rates=[6, 12, 18])

        # ===== Decoder =====
        self.up1 = nn.ConvTranspose2d(1024, 256, 2, 2) 
        self.att1 = AttentionGate(256, 1024, 256)
        self.res1 = ResidualConv(256 + 1024, 256)  

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 512, 128)
        self.res2 = ResidualConv(128 + 512, 128)   

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att3 = AttentionGate(64, 256, 64)
        self.res3 = ResidualConv(64 + 256, 64)    

        self.up4 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.att4 = AttentionGate(64, 64, 32)
        self.res4 = ResidualConv(64 + 64, 64)      

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        # ===== Encoder =====
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        b  = self.layer4(s4)
        b = self.aspp(b)  
        # ===== Decoder =====
        x = self.up1(b)
        x = torch.cat([x, self.att1(x, s4)], dim=1)
        x = self.res1(x)

        x = self.up2(x)
        x = torch.cat([x, self.att2(x, s3)], dim=1)
        x = self.res2(x)

        x = self.up3(x)
        x = torch.cat([x, self.att3(x, s2)], dim=1)
        x = self.res3(x)

        x = self.up4(x)
        x = torch.cat([x, self.att4(x, s1)], dim=1)
        x = self.res4(x)

        x = self.outc(x)

        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout2d(p=0.25),
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        # Increased out_channels to 1024 to accommodate ResNeXt diversity
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            *[nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for r in rates],
            nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        res = []
        for stage in self.stages:
            if isinstance(stage[0], nn.AdaptiveAvgPool2d):
                y = stage(x)
                y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
                res.append(y)
            else:
                res.append(stage(x))
        return self.bottleneck(torch.cat(res, dim=1))