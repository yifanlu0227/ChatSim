import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size = 3,
        stride: int = 1,
        act: str = 'relu',
        use_bn: bool = True
        ):
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)

        if inplanes != planes or stride != 1:
            self.downsample = nn.Sequential(
                    conv1x1(inplanes, planes, stride),
                    self.bn3,
                )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
    

class UpBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size = 3,
        stride: int = 1,
        act: str = 'relu',
        use_bn: bool = True
        ):
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, output_padding=kernel_size//2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'selu':
            self.act = nn.SELU(inplace=True)
        elif act == 'elu':
            self.act = nn.ELU(inplace=True)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out


def build_layer(inplane, plane, kernel_size, stride, block_num, act='relu', use_bn=True):
    module_list = []

    module_list.append(BasicBlock(inplane, plane, kernel_size=kernel_size, stride=stride, act=act, use_bn=use_bn))
    for _ in range(1, block_num):
        module_list.append(BasicBlock(plane, plane, kernel_size=kernel_size, act=act, use_bn=use_bn))

    return nn.Sequential(*module_list)

def build_up_layer(inplane, plane, kernel_size, stride, block_num, act='relu', use_bn=True):
    """
    Here stride refers to upsampling stride
    """
    module_list = []

    module_list.append(UpBasicBlock(inplane, plane, kernel_size=kernel_size, stride=stride, act=act, use_bn=use_bn))
    for _ in range(1, block_num):
        module_list.append(BasicBlock(plane, plane, kernel_size=kernel_size, act=act, use_bn=use_bn))

    return nn.Sequential(*module_list)
