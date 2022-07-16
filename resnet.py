import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DownSampleA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownSampleA, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x,x.mul(0)), 1)

class ResNetBasicBlock(nn.Module):
    expansion = 1
    """
    ResNet BasicBlock from the paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.featureSize = 64

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in the paper:
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, block, depth, num_classes, channels = 3):
        """ Constructor
        Args:
            block: class for the block to use for depth layers
            depth: number of layers in the network
            num_classes: number of classes
            channels: number of channels in the image, 3 for RGB
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2 eg 20, 32, 44, 56, 68, 80'
        n = (depth - 2) // 6

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.inplanes = 16
        self.stage1 = self._make_layer(block, 16, n, 1)
        self.stage2 = self._make_layer(block, 32, n, 2)
        self.stage3 = self._make_layer(block, 64, n, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownSampleA(self.inplanes, planes * block.expansion, stride)
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
        
    def resnet20(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 20, num_classes)
    
    def resnet10mnist(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 10, num_classes, 1)
    
    def resnet20mnist(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 20, num_classes, 1)
    
    def resnet32mnist(num_classes=10, channels=1):
        return CifarResNet(ResNetBasicBlock, 32, num_classes, channels)

    def resnet32(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 32, num_classes)

    def resnet44(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 44, num_classes)
    
    def resnet56(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 56, num_classes)
    
    def resnet110(num_classes=10):
        return CifarResNet(ResNetBasicBlock, 110, num_classes)