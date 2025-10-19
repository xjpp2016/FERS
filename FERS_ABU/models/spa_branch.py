import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights, ResNet18_Weights
import torch.nn.functional as F
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights

def init_weights(m, mode="uniform", **kwargs):
    """
    General weight initialization function:
    - mode: options include
        "kaiming_uniform", "kaiming_normal",
        "xavier_uniform", "xavier_normal",
        "normal", "uniform", "orthogonal", "sparse"
    - kwargs: additional parameters like mean, std, a, sparsity, gain
    """

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if mode == "kaiming_uniform":
            init.kaiming_uniform_(m.weight, a=kwargs.get("a", 0), nonlinearity="relu")
        elif mode == "kaiming_normal":
            init.kaiming_normal_(m.weight, a=kwargs.get("a", 0), nonlinearity="relu")
        elif mode == "xavier_uniform":
            init.xavier_uniform_(m.weight, gain=kwargs.get("gain", 1.0))
        elif mode == "xavier_normal":
            init.xavier_normal_(m.weight, gain=kwargs.get("gain", 1.0))
        elif mode == "normal":
            init.normal_(m.weight, mean=kwargs.get("mean", 0.0), std=kwargs.get("std", 0.02))
        elif mode == "uniform":
            init.uniform_(m.weight, a=kwargs.get("a", -0.1), b=kwargs.get("b", 0.1))
        elif mode == "orthogonal":
            init.orthogonal_(m.weight, gain=kwargs.get("gain", 1.0))
        elif mode == "sparse":
            init.sparse_(m.weight, sparsity=kwargs.get("sparsity", 0.1), std=kwargs.get("std", 0.01))
        else:
            raise ValueError(f"Unknown init mode: {mode}")

        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1.0)
        init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.Embedding):
        dim = m.embedding_dim
        bound = 1 / dim**0.5
        init.uniform_(m.weight, -bound, bound)


class Conv2RGB(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(Conv2RGB, self).__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.relu = nn.ReLU()
        self.hidden = nn.Conv2d(256, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        return x

# -------------------------
# Channel Attention Module
# -------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class LightSPP(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 pool_sizes=[1, 2, 4, 8], use_dilated=False):
        super(LightSPP, self).__init__()
        self.pool_sizes = pool_sizes
        self.use_dilated = use_dilated

        # Multi-scale branches
        self.branches = nn.ModuleList()
        for size in pool_sizes:
            self.branches.append(
                nn.AdaptiveAvgPool2d(size)
            )

        mid_channels = max(in_channels // 2, 16)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * len(pool_sizes), mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(out_channels)

        self.apply(init_weights)

    def forward(self, x, target_size):
        feats = []
        for op in self.branches:
            f = op(x)
            f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            feats.append(f)

        out = torch.cat(feats, dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out

# -------------------------
# Spatial Feature Enhancement Main Network
# -------------------------
class HyperSpatialResNet(nn.Module):
    def __init__(self, input_channels, use_dilated=False):
        super(HyperSpatialResNet, self).__init__()
        
        # Input channels → 3 channels (required for ResNet pretrained weights)
        self.initial_conv = Conv2RGB(input_channels)
        

        resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        # Load pretrained ResNet34
        self.get_features = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,  # 64 channels
            resnet34.layer2,  # 128 channels
            resnet34.layer3,  # 256 channels
        )

        # Freeze ResNet parameters (only freeze backbone, not initial and custom layers)
        for param in self.get_features.parameters():
            param.requires_grad = False
            
        self.spp = LightSPP(in_channels=256, 
                            out_channels=input_channels,
                            pool_sizes=[1, 2, 4])
        
    def forward(self, x):
        input_size = x.size()[-2:]   # Input H, W
        false_RGB = self.initial_conv(x)
        features = self.get_features(false_RGB)
        output = self.spp(features, input_size)
        return output, false_RGB
    
