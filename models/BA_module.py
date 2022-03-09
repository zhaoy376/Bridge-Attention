from torch import nn
import torch
from functools import reduce
import torch.nn.functional as F
import math

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class BA_module_resnet(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=16):  
        super(BA_module_resnet, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
                nn.Linear(cur_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction)
            )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer, Gate, pre_block=None):
        b, cur_c, _, _ = cur_layer.size()

        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights

class BA_module_mobilenetv3(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=6):  
        super(BA_module_mobilenetv3, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction),
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
            nn.Linear(cur_channel, cur_channel // reduction, bias=False),
            nn.BatchNorm1d(cur_channel // reduction),
        )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            h_sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.size()
        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))

        fusion = cur_fusion + sum(pre_fusions)
        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights

class BA_module_efficientnet(nn.Module):
    def __init__(self, pre_channels, cur_channel, reduction=6):
        super(BA_module_efficientnet, self).__init__()
        self.pre_fusions = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias=False),
                nn.BatchNorm1d(cur_channel // reduction),
            )
                for pre_channel in pre_channels]
        )

        self.cur_fusion = nn.Sequential(
            nn.Linear(cur_channel, cur_channel // reduction, bias=False),
            nn.BatchNorm1d(cur_channel // reduction),
        )

        self.generation = nn.Sequential(
            MemoryEfficientSwish(),
            nn.Linear(cur_channel // reduction, cur_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.size()
        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))

        fusion = cur_fusion + sum(pre_fusions)
        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights
