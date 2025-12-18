import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.model.layers import *
from src.model.conv_cdc3d import CDC_ST, CDC_Dynamic, TemporalDiffConv3d


class ResNet3d(nn.Module):
    def __init__(self, block, layers, num_classes=400, conv_type='vanilla', in_channels=3):
        super(ResNet3d, self).__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        if conv_type == 'vanilla':
            conv_makers = [Conv3DSimple] * 4
        elif conv_type == 'cdc':
            conv_makers = [CDC_ST] * 4
        elif conv_type == 'dynamic':
            conv_makers = [CDC_Dynamic] * 4
        elif conv_type == 'temporal':
            conv_makers = [TemporalDiffConv3d] * 4

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

class ResNet3d_fc(nn.Module):
    def __init__(self, block, layers, num_classes=400, conv_type='vanilla'):
        super(ResNet3d_fc, self).__init__()
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        if conv_type == 'vanilla':
            conv_makers = [Conv3DSimple] * 4
        elif conv_type == 'cdc':
            conv_makers = [CDC_ST] * 4
        elif conv_type == 'dynamic':
            conv_makers = [CDC_Dynamic] * 4

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


class Four_ResNet3d(nn.Module):
    def __init__(self, block, layers, num_classes=400, conv_type='vanilla'):
        super().__init__()
        self.wavelet = DWT3d()
        self.r3d_ll = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_lh = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hl = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hh = ResNet3d(block, layers, num_classes, conv_type)
        self.fuse = nn.Sequential(
            nn.Conv3d(512 * 4 * block.expansion, 512 * block.expansion, kernel_size=1,
                      bias=False),
            nn.BatchNorm3d(512 * block.expansion),
            nn.ReLU(inplace=True),
            nn.Conv3d(512 * block.expansion, 512 * block.expansion, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm3d(512 * block.expansion),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x_ll, x_lh, x_hl, x_hh = self.wavelet(x)
        x_ll = self.r3d_ll(x_ll)
        x_lh = self.r3d_lh(x_lh)
        x_hl = self.r3d_hl(x_hl)
        x_hh = self.r3d_hh(x_hh)
        f_cat = torch.cat((x_ll, x_lh, x_hl, x_hh), dim=1)
        x_out = self.fuse(f_cat)
        x_out = self.avgpool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc(x_out)

        return x_out
    
    
if __name__ == '__main__':
    x = torch.randn((1, 3, 16, 224, 224))
    model = ResNet3d_fc(BasicBlock, [2, 2, 2, 2], conv_type='vanilla')
    out = model(x)
    print(out.shape)
