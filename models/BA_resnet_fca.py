import torch.nn as nn
from torch.hub import load_state_dict_from_url
#from torchvision.models import ResNet
from models.BA_module import BA_module_resnet
from models.DCT_extration import MultiSpectralAttentionLayer
import torch


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, Gate):
        super(BABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ba = BA_module_resnet([planes], planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)
        self.Gate = Gate

        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.feature_extraction1 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction,
                                                               freq_sel_method = 'top16')
        self.feature_extraction2 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                               freq_sel_method='top16')

    def forward(self, x):
        if isinstance(x, list):
            x, prev = x[0], x[1]
        else:
            prev = None
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        F1 = self.feature_extraction1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        F2 = self.feature_extraction2(out)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, Gate):
        super(BABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ba = BA_module_resnet([planes, planes], 4*planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.feature_extraction = nn.AdaptiveAvgPool2d(1)
        self.Gate = Gate

        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.feature_extraction1 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes],  reduction=reduction,
                                                               freq_sel_method = 'top16')
        self.feature_extraction2 = MultiSpectralAttentionLayer(planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                               freq_sel_method='top16')
        self.feature_extraction3 = MultiSpectralAttentionLayer(4 * planes, c2wh[planes], c2wh[planes], reduction=reduction,
                                                               freq_sel_method='top16')

    def forward(self, x):
        if isinstance(x, list):
            x, prev = x[0], x[1]
        else:
            prev = None
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        F1 = self.feature_extraction1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        F2 = self.feature_extraction2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        F3 = self.feature_extraction3(out)
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

class BANet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, reduction=16):
        super(BANet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.reduction = reduction
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.expand = 1
        if 'Bottleneck' in str(block):
            self.expand = 4
        self.gate1 = self._make_gate(64*self.expand)
        self.gate2 = self._make_gate(128*self.expand)
        self.gate3 = self._make_gate(256*self.expand)
        self.gate4 = self._make_gate(512*self.expand)

        self.layer1 = self._make_layer(block, 64, layers[0], Gate=self.gate1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], Gate=self.gate2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], Gate=self.gate3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], Gate=self.gate4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BABottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_gate(self, planes):
        #previous block-gate
        inplanes = 2 * (planes // self.reduction)
        outplanes = planes // self.reduction
        Gate = nn.Sequential(
            nn.Linear(inplanes, outplanes, bias=False),
            nn.Sigmoid()
        )
        return Gate

    def _make_layer(self, block, planes, blocks, Gate, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, Gate=Gate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, Gate=Gate))

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

        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ba_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BANet(BABasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def ba_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BANet(BABasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def ba_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BANet(BABottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model


def ba_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BANet(BABottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def ba_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BANet(BABottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


