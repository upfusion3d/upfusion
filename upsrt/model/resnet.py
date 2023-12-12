import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNetConv(nn.Module):
    def __init__(self, n_blocks=3, use_feature_pyramid=False, num_patches_x=None, num_patches_y=None):
        super(ResNetConv, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.n_blocks = n_blocks
        self.use_feature_pyramid = use_feature_pyramid
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        out = x

        if n_blocks >= 1:
            x = self.resnet.layer1(x)  # (B, C, H/2, W/2)
            if self.use_feature_pyramid:
                out = F.interpolate(x, size=(self.num_patches_y, self.num_patches_x), mode='bilinear')  # (B,C1,Py,Px)
            else:
                out = x
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
            if self.use_feature_pyramid:
                x = F.interpolate(x, size=(self.num_patches_y, self.num_patches_x), mode='bilinear')  # (B,C2,Py,Px)
                out = torch.cat([out, x], dim=1)  # (B,C1+C2,Py,Px)
            else:
                out = x
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
            if self.use_feature_pyramid:
                x = F.interpolate(x, size=(self.num_patches_y, self.num_patches_x), mode='bilinear')  # (B,C3,Py,Px)
                out = torch.cat([out, x], dim=1)  # (B,C1+C2+C3,Py,Px)
            else:
                out = x
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
            if self.use_feature_pyramid:
                x = F.interpolate(x, size=(self.num_patches_y, self.num_patches_x), mode='bilinear')  # (B,C,Py,Px)
                out = torch.cat([out, x], dim=1)  # (B,4C,Py,Px)
            else:
                out = x

        return out
