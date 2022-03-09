import torch.nn as nn
import torch
import math
from models.BA_module import BA_module_resnet
__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152', 'ba_resnext18', 'ba_resnext34','ba_resnext50', 'ba_resnext101',
           'ba_resnext152']

def conv3x3(in_planes, out_planes, groups, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=32):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group, )
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class BABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, reduction=16, Gate=None):
        super(BABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

        self.ba = BA_module_resnet([planes*2], planes*2, reduction)
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)
        self.Gate = Gate

    def forward(self, x):
        if isinstance(x, list):
            x, prev = x[0], x[1]
        else:
            prev = None
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        F1 = self.feature_extraction(out)

        out = self.conv2(out)
        out = self.bn2(out)
        F2 = self.feature_extraction(out)
        if self.downsample is not None:
            att, fusion = self.ba([F1], F2, self.Gate)
        else:
            att, fusion = self.ba([F1], F2, self.Gate, prev)
        out = out * att

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return [out, fusion]

class BABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, reduction=16, Gate=None):
        super(BABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.ba = BA_module_resnet([planes*2, planes*2], 4 * planes, reduction)
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)
        self.Gate = Gate

    def forward(self, x):
        if isinstance(x, list):
            x, prev = x[0], x[1]
        else:
            prev = None
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        F1 = self.feature_extraction(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        F2 = self.feature_extraction(out)

        out = self.conv3(out)
        out = self.bn3(out)
        F3 = self.feature_extraction(out)
        if self.downsample is not None:
            att, fusion = self.ba([F1, F2], F3, self.Gate)
        else:
            att, fusion = self.ba([F1, F2], F3, self.Gate, prev)
        out = out * att

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return [out, fusion]

class BAResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=32, reduction=16):
        self.inplanes = 64
        self.reduction =reduction
        super(BAResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.expand = 1
        if 'Bottleneck' in str(block):
            self.expand = 4
        self.gate1 = self._make_gate(64 * self.expand)
        self.gate2 = self._make_gate(128 * self.expand)
        self.gate3 = self._make_gate(256 * self.expand)
        self.gate4 = self._make_gate(512 * self.expand)

        self.layer1 = self._make_layer(block, 64, layers[0], num_group, Gate=self.gate1)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2, Gate=self.gate2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2, Gate=self.gate3)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2, Gate=self.gate4)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, Gate, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group, Gate=Gate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group, Gate=Gate))

        return nn.Sequential(*layers)

    def _make_gate(self, planes):
        inplanes = 2 * (planes // self.reduction)
        outplanes = planes // self.reduction
        Gate = nn.Sequential(
            nn.Linear(inplanes, outplanes, bias=False),
            nn.Sigmoid()
        )
        # inside block-gate and previous block-gate
        return Gate

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ba_resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = BAResNeXt(BABasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def ba_resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = BAResNeXt(BABasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ba_resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = BAResNeXt(BABottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ba_resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    model = BAResNeXt(BABottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ba_resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = BAResNeXt(BABottleneck, [3, 8, 36, 3], **kwargs)
    return model