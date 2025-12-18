from functools import partial
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.model.layers import *
from src.model.resnet3d import ResNet3d


class ResNet3d_pro(nn.Module):
    def __init__(self, block, layers, num_classes=400, conv_type='vanilla', max_or_avg='max'):
        super().__init__()
        self.dwt3d = DWT3d()
        self.dwtTime = DWTtime()
        self.r3d_ll = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_lh = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hl = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hh = ResNet3d(block, layers, num_classes, conv_type)

        # 权重生成模块
        in_channels = 512 * block.expansion
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels * 4, in_channels * 4 // 16),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4 // 16, 4),  # 输出4个权重
            nn.Sigmoid()  # 限制权重在0-1之间
        )

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
        self.dhp1 = nn.Sequential(nn.Conv3d(64 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp2 = nn.Sequential(nn.Conv3d(128 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp3 = nn.Sequential(nn.Conv3d(256 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp4 = nn.Sequential(nn.Conv3d(512 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))

        self.pool2_h = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.pool3_h = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.pool4_h = nn.MaxPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1))

        self.pool2_l = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.pool3_l = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.pool4_l = nn.MaxPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1))

    def _dwt_time(self, lh, hl, hh, is_secret=None):
        lh_l, lh_h = self.dwtTime(lh)
        hl_l, hl_h = self.dwtTime(hl)
        hh_l, hh_h = self.dwtTime(hh)

        if is_secret == None or is_secret == 1:
            return lh_l, lh_h, hl_l, hl_h, hh_l, hh_h
        elif is_secret == 2:
            return self.pool2_l(lh_l), self.pool2_h(lh_h), self.pool2_l(hl_l), self.pool2_h(hl_h), self.pool2_l(
                hh_l), self.pool2_h(hh_h)
        elif is_secret == 3:
            return self.pool3_l(lh_l), self.pool3_h(lh_h), self.pool3_l(hl_l), self.pool3_h(hl_h), self.pool3_l(
                hh_l), self.pool3_h(hh_h)
        elif is_secret == 4:
            return self.pool4_l(lh_l), self.pool4_h(lh_h), self.pool4_l(hl_l), self.pool4_h(hl_h), self.pool4_l(
                hh_l), self.pool4_h(hh_h)

    def forward(self, is_train=False, **kwargs):
        x = kwargs['stego']
        x_ll, x_lh, x_hl, x_hh = self.dwt3d(x)
        if is_train:
            x_sec = kwargs['secret']
            x_sec_ll, _, _, _ = self.dwt3d(x_sec)
            x_sec_ll_1, x_sec_lh_1, x_sec_hl_1, x_sec_hh_1 = self.dwt3d(x_sec_ll)
            x_sec_ll_2, x_sec_lh_2, x_sec_hl_2, x_sec_hh_2 = self.dwt3d(x_sec_ll_1)
            x_sec_ll_3, x_sec_lh_3, x_sec_hl_3, x_sec_hh_3 = self.dwt3d(x_sec_ll_2)
            _, x_sec_lh_4, x_sec_hl_4, x_sec_hh_4 = self.dwt3d(x_sec_ll_3)

            _, _, _, x_sec_lh_h_1, x_sec_hl_h_1, x_sec_hh_h_1 = self._dwt_time(x_sec_lh_1, x_sec_hl_1, x_sec_hh_1,
                                                                               is_secret=1)
            _, _, _, x_sec_lh_h_2, x_sec_hl_h_2, x_sec_hh_h_2 = self._dwt_time(x_sec_lh_2, x_sec_hl_2, x_sec_hh_2,
                                                                               is_secret=2)
            _, _, _, x_sec_lh_h_3, x_sec_hl_h_3, x_sec_hh_h_3 = self._dwt_time(x_sec_lh_3, x_sec_hl_3, x_sec_hh_3,
                                                                               is_secret=3)
            _, _, _, x_sec_lh_h_4, x_sec_hl_h_4, x_sec_hh_h_4 = self._dwt_time(x_sec_lh_4, x_sec_hl_4, x_sec_hh_4,
                                                                               is_secret=4)

        x_ll = self.r3d_ll.stem(x_ll)
        x_lh = self.r3d_lh.stem(x_lh)
        x_hl = self.r3d_hl.stem(x_hl)
        x_hh = self.r3d_hh.stem(x_hh)

        x_ll_l1 = self.r3d_ll.layer1(x_ll)
        x_lh_l1 = self.r3d_lh.layer1(x_lh)
        x_hl_l1 = self.r3d_hl.layer1(x_hl)
        x_hh_l1 = self.r3d_hh.layer1(x_hh)

        x_ll_l2 = self.r3d_ll.layer2(x_ll_l1)
        x_lh_l2 = self.r3d_lh.layer2(x_lh_l1)
        x_hl_l2 = self.r3d_hl.layer2(x_hl_l1)
        x_hh_l2 = self.r3d_hh.layer2(x_hh_l1)

        x_ll_l3 = self.r3d_ll.layer3(x_ll_l2)
        x_lh_l3 = self.r3d_lh.layer3(x_lh_l2)
        x_hl_l3 = self.r3d_hl.layer3(x_hl_l2)
        x_hh_l3 = self.r3d_hh.layer3(x_hh_l2)

        x_ll_l4 = self.r3d_ll.layer4(x_ll_l3)
        x_lh_l4 = self.r3d_lh.layer4(x_lh_l3)
        x_hl_l4 = self.r3d_hl.layer4(x_hl_l3)
        x_hh_l4 = self.r3d_hh.layer4(x_hh_l3)

        gap_ll = self.gap(x_ll_l4).flatten(1)
        gap_lh = self.gap(x_lh_l4).flatten(1)
        gap_hl = self.gap(x_hl_l4).flatten(1)
        gap_hh = self.gap(x_hh_l4).flatten(1)

        weights = self.weight_generator(torch.cat([gap_ll, gap_lh, gap_hl, gap_hh], dim=1))
        w_ll, w_lh, w_hl, w_hh = weights.chunk(4, dim=1)  # 分割成4个权重

        w_ll = w_ll.view(-1, 1, 1, 1, 1)
        w_lh = w_lh.view(-1, 1, 1, 1, 1)
        w_hl = w_hl.view(-1, 1, 1, 1, 1)
        w_hh = w_hh.view(-1, 1, 1, 1, 1)

        x_out = self.fuse(torch.cat((x_ll_l4 * w_ll, x_lh_l4 * w_lh, x_hl_l4 * w_hl, x_hh_l4 * w_hh), dim=1))
        x_out = self.avgpool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc(x_out)

        if is_train:
            x_lh_l1 = self.dhp1(x_lh_l1)
            x_hl_l1 = self.dhp1(x_hl_l1)
            x_hh_l1 = self.dhp1(x_hh_l1)
            x_lh_l2 = self.dhp2(x_lh_l2)
            x_hl_l2 = self.dhp2(x_hl_l2)
            x_hh_l2 = self.dhp2(x_hh_l2)
            x_lh_l3 = self.dhp3(x_lh_l3)
            x_hl_l3 = self.dhp3(x_hl_l3)
            x_hh_l3 = self.dhp3(x_hh_l3)
            x_lh_l4 = self.dhp4(x_lh_l4)
            x_hl_l4 = self.dhp4(x_hl_l4)
            x_hh_l4 = self.dhp4(x_hh_l4)

            _, _, _, x_lh_h_1, x_hl_h_1, x_hh_h_1 = self._dwt_time(x_lh_l1, x_hl_l1, x_hh_l1)
            _, _, _, x_lh_h_2, x_hl_h_2, x_hh_h_2 = self._dwt_time(x_lh_l2, x_hl_l2, x_hh_l2)
            _, _, _, x_lh_h_3, x_hl_h_3, x_hh_h_3 = self._dwt_time(x_lh_l3, x_hl_l3, x_hh_l3)
            _, _, _, x_lh_h_4, x_hl_h_4, x_hh_h_4 = self._dwt_time(x_lh_l4, x_hl_l4, x_hh_l4)

            x_sec_lh_2 = self.pool2_h(x_sec_lh_2)
            x_sec_hl_2 = self.pool2_h(x_sec_hl_2)
            x_sec_hh_2 = self.pool2_h(x_sec_hh_2)
            x_sec_lh_3 = self.pool3_h(x_sec_lh_3)
            x_sec_hl_3 = self.pool3_h(x_sec_hl_3)
            x_sec_hh_3 = self.pool3_h(x_sec_hh_3)
            x_sec_lh_4 = self.pool4_h(x_sec_lh_4)
            x_sec_hl_4 = self.pool4_h(x_sec_hl_4)
            x_sec_hh_4 = self.pool4_h(x_sec_hh_4)

            return [[(torch.cat((x_lh_l1, x_hl_l1, x_hh_l1), dim=1),
                      torch.cat((x_sec_lh_1, x_sec_hl_1, x_sec_hh_1), dim=1)),
                     (torch.cat((x_lh_l2, x_hl_l2, x_hh_l2), dim=1),
                      torch.cat((x_sec_lh_2, x_sec_hl_2, x_sec_hh_2), dim=1)),
                     (torch.cat((x_lh_l3, x_hl_l3, x_hh_l3), dim=1),
                      torch.cat((x_sec_lh_3, x_sec_hl_3, x_sec_hh_3), dim=1)),
                     (torch.cat((x_lh_l4, x_hl_l4, x_hh_l4), dim=1),
                      torch.cat((x_sec_lh_4, x_sec_hl_4, x_sec_hh_4), dim=1))],
                    [(torch.cat((x_lh_h_1, x_hl_h_1, x_hh_h_1), dim=1),
                      torch.cat((x_sec_lh_h_1, x_sec_hl_h_1, x_sec_hh_h_1),
                                dim=1)),
                     (torch.cat((x_lh_h_2, x_hl_h_2, x_hh_h_2), dim=1),
                      torch.cat((x_sec_lh_h_2, x_sec_hl_h_2, x_sec_hh_h_2),
                                dim=1)),
                     (torch.cat((x_lh_h_3, x_hl_h_3, x_hh_h_3), dim=1),
                      torch.cat((x_sec_lh_h_3, x_sec_hl_h_3, x_sec_hh_h_3),
                                dim=1)),
                     (torch.cat((x_lh_h_4, x_hl_h_4, x_hh_h_4), dim=1),
                      torch.cat((x_sec_lh_h_4, x_sec_hl_h_4, x_sec_hh_h_4),
                                dim=1))],
                    x_out]

        else:
            return x_out


class ResNet3d_pro_TA(nn.Module):
    def __init__(self, block, layers, num_classes=400, conv_type='vanilla', theta=0.2):
        super().__init__()
        self.dwt3d = DWT3d()
        self.dwtTime = DWTtime()
        self.r3d_ll = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_lh = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hl = ResNet3d(block, layers, num_classes, conv_type)
        self.r3d_hh = ResNet3d(block, layers, num_classes, conv_type)

        in_channels = 512 * block.expansion
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels * 4, in_channels * 4 // 16),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4 // 16, 4),  # 输出4个权重
            nn.Sigmoid()  # 限制权重在0-1之间
        )
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

        self.ca1 = CrossTemporalAttention(64 * block.expansion, with_sa=True, max_temporal_len=16, theta=theta)
        self.ca2 = CrossTemporalAttention(128 * block.expansion, with_sa=True, max_temporal_len=8, theta=theta)
        self.ca3 = CrossTemporalAttention(256 * block.expansion, with_sa=True, max_temporal_len=4, theta=theta)
        self.ca4 = CrossTemporalAttention(512 * block.expansion, with_sa=True, max_temporal_len=2, theta=theta)

        self.dhp1 = nn.Sequential(nn.Conv3d(64 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp2 = nn.Sequential(nn.Conv3d(128 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp3 = nn.Sequential(nn.Conv3d(256 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))
        self.dhp4 = nn.Sequential(nn.Conv3d(512 * block.expansion, 3, kernel_size=1, bias=False),
                                  nn.BatchNorm3d(3),
                                  nn.ReLU(inplace=True))

        self.pool2_h = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.pool3_h = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.pool4_h = nn.MaxPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1))

        self.pool2_l = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.pool3_l = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.pool4_l = nn.MaxPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1))

    def _dwt_time(self, lh, hl, hh, is_secret=None):
        lh_l, lh_h = self.dwtTime(lh)
        hl_l, hl_h = self.dwtTime(hl)
        hh_l, hh_h = self.dwtTime(hh)

        if is_secret == None or is_secret == 1:
            return lh_l, lh_h, hl_l, hl_h, hh_l, hh_h
        elif is_secret == 2:
            return self.pool2_l(lh_l), self.pool2_h(lh_h), self.pool2_l(hl_l), self.pool2_h(hl_h), self.pool2_l(
                hh_l), self.pool2_h(hh_h)
        elif is_secret == 3:
            return self.pool3_l(lh_l), self.pool3_h(lh_h), self.pool3_l(hl_l), self.pool3_h(hl_h), self.pool3_l(
                hh_l), self.pool3_h(hh_h)
        elif is_secret == 4:
            return self.pool4_l(lh_l), self.pool4_h(lh_h), self.pool4_l(hl_l), self.pool4_h(hl_h), self.pool4_l(
                hh_l), self.pool4_h(hh_h)

    def forward(self, is_train=False, **kwargs):
        x = kwargs['stego']
        x_ll, x_lh, x_hl, x_hh = self.dwt3d(x)
        if is_train:
            x_sec = kwargs['secret']
            x_sec_ll, _, _, _ = self.dwt3d(x_sec)
            x_sec_ll_1, x_sec_lh_1, x_sec_hl_1, x_sec_hh_1 = self.dwt3d(x_sec_ll)
            x_sec_ll_2, x_sec_lh_2, x_sec_hl_2, x_sec_hh_2 = self.dwt3d(x_sec_ll_1)
            x_sec_ll_3, x_sec_lh_3, x_sec_hl_3, x_sec_hh_3 = self.dwt3d(x_sec_ll_2)
            _, x_sec_lh_4, x_sec_hl_4, x_sec_hh_4 = self.dwt3d(x_sec_ll_3)

            _, _, _, x_sec_lh_h_1, x_sec_hl_h_1, x_sec_hh_h_1 = self._dwt_time(x_sec_lh_1, x_sec_hl_1, x_sec_hh_1, is_secret=1)
            _, _, _, x_sec_lh_h_2, x_sec_hl_h_2, x_sec_hh_h_2 = self._dwt_time(x_sec_lh_2, x_sec_hl_2, x_sec_hh_2, is_secret=2)
            _, _, _, x_sec_lh_h_3, x_sec_hl_h_3, x_sec_hh_h_3 = self._dwt_time(x_sec_lh_3, x_sec_hl_3, x_sec_hh_3, is_secret=3)
            _, _, _, x_sec_lh_h_4, x_sec_hl_h_4, x_sec_hh_h_4 = self._dwt_time(x_sec_lh_4, x_sec_hl_4, x_sec_hh_4, is_secret=4)

        x_ll = self.r3d_ll.stem(x_ll)
        x_lh = self.r3d_lh.stem(x_lh)
        x_hl = self.r3d_hl.stem(x_hl)
        x_hh = self.r3d_hh.stem(x_hh)

        x_ll_l1 = self.r3d_ll.layer1(x_ll)
        x_lh_l1 = self.r3d_lh.layer1(x_lh)
        x_hl_l1 = self.r3d_hl.layer1(x_hl)
        x_hh_l1 = self.r3d_hh.layer1(x_hh)
        # x_lh_l1 = self.ca1(x_lh_l1, x_ll_l1)
        # x_hl_l1 = self.ca1(x_hl_l1, x_ll_l1)
        # x_hh_l1 = self.ca1(x_hh_l1, x_ll_l1)
        x_lh_l1 = self.ca1(x_ll_l1, x_lh_l1)
        x_hl_l1 = self.ca1(x_ll_l1, x_hl_l1)
        x_hh_l1 = self.ca1(x_ll_l1, x_hh_l1)

        x_ll_l2 = self.r3d_ll.layer2(x_ll_l1)
        x_lh_l2 = self.r3d_lh.layer2(x_lh_l1)
        x_hl_l2 = self.r3d_hl.layer2(x_hl_l1)
        x_hh_l2 = self.r3d_hh.layer2(x_hh_l1)
        # x_lh_l2 = self.ca2(x_lh_l2, x_ll_l2)
        # x_hl_l2 = self.ca2(x_hl_l2, x_ll_l2)
        # x_hh_l2 = self.ca2(x_hh_l2, x_ll_l2)
        x_lh_l2 = self.ca2(x_ll_l2, x_lh_l2)
        x_hl_l2 = self.ca2(x_ll_l2, x_hl_l2)
        x_hh_l2 = self.ca2(x_ll_l2, x_hh_l2)

        x_ll_l3 = self.r3d_ll.layer3(x_ll_l2)
        x_lh_l3 = self.r3d_lh.layer3(x_lh_l2)
        x_hl_l3 = self.r3d_hl.layer3(x_hl_l2)
        x_hh_l3 = self.r3d_hh.layer3(x_hh_l2)
        # x_lh_l3 = self.ca3(x_lh_l3, x_ll_l3)
        # x_hl_l3 = self.ca3(x_hl_l3, x_ll_l3)
        # x_hh_l3 = self.ca3(x_hh_l3, x_ll_l3)
        x_lh_l3 = self.ca3(x_ll_l3, x_lh_l3)
        x_hl_l3 = self.ca3(x_ll_l3, x_hl_l3)
        x_hh_l3 = self.ca3(x_ll_l3, x_hh_l3)

        x_ll_l4 = self.r3d_ll.layer4(x_ll_l3)
        x_lh_l4 = self.r3d_lh.layer4(x_lh_l3)
        x_hl_l4 = self.r3d_hl.layer4(x_hl_l3)
        x_hh_l4 = self.r3d_hh.layer4(x_hh_l3)
        # x_lh_l4 = self.ca4(x_lh_l4, x_ll_l4)
        # x_hl_l4 = self.ca4(x_hl_l4, x_ll_l4)
        # x_hh_l4 = self.ca4(x_hh_l4, x_ll_l4)
        x_lh_l4 = self.ca4(x_ll_l4, x_lh_l4)
        x_hl_l4 = self.ca4(x_ll_l4, x_hl_l4)
        x_hh_l4 = self.ca4(x_ll_l4, x_hh_l4)

        gap_ll = self.gap(x_ll_l4).flatten(1)
        gap_lh = self.gap(x_lh_l4).flatten(1)
        gap_hl = self.gap(x_hl_l4).flatten(1)
        gap_hh = self.gap(x_hh_l4).flatten(1)

        weights = self.weight_generator(torch.cat([gap_ll, gap_lh, gap_hl, gap_hh], dim=1))
        w_ll, w_lh, w_hl, w_hh = weights.chunk(4, dim=1)  # 分割成4个权重

        w_ll = w_ll.view(-1, 1, 1, 1, 1)
        w_lh = w_lh.view(-1, 1, 1, 1, 1)
        w_hl = w_hl.view(-1, 1, 1, 1, 1)
        w_hh = w_hh.view(-1, 1, 1, 1, 1)

        x_out = self.fuse(torch.cat((x_ll_l4 * w_ll, x_lh_l4 * w_lh, x_hl_l4 * w_hl, x_hh_l4 * w_hh), dim=1))
        x_out = self.avgpool(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc(x_out)

        if is_train:
            x_lh_l1 = self.dhp1(x_lh_l1)
            x_hl_l1 = self.dhp1(x_hl_l1)
            x_hh_l1 = self.dhp1(x_hh_l1)
            x_lh_l2 = self.dhp2(x_lh_l2)
            x_hl_l2 = self.dhp2(x_hl_l2)
            x_hh_l2 = self.dhp2(x_hh_l2)
            x_lh_l3 = self.dhp3(x_lh_l3)
            x_hl_l3 = self.dhp3(x_hl_l3)
            x_hh_l3 = self.dhp3(x_hh_l3)
            x_lh_l4 = self.dhp4(x_lh_l4)
            x_hl_l4 = self.dhp4(x_hl_l4)
            x_hh_l4 = self.dhp4(x_hh_l4)

            _, _, _, x_lh_h_1, x_hl_h_1, x_hh_h_1 = self._dwt_time(x_lh_l1, x_hl_l1, x_hh_l1)
            _, _, _, x_lh_h_2, x_hl_h_2, x_hh_h_2 = self._dwt_time(x_lh_l2, x_hl_l2, x_hh_l2)
            _, _, _, x_lh_h_3, x_hl_h_3, x_hh_h_3 = self._dwt_time(x_lh_l3, x_hl_l3, x_hh_l3)
            _, _, _, x_lh_h_4, x_hl_h_4, x_hh_h_4 = self._dwt_time(x_lh_l4, x_hl_l4, x_hh_l4)

            x_sec_lh_2 = self.pool2_h(x_sec_lh_2)
            x_sec_hl_2 = self.pool2_h(x_sec_hl_2)
            x_sec_hh_2 = self.pool2_h(x_sec_hh_2)
            x_sec_lh_3 = self.pool3_h(x_sec_lh_3)
            x_sec_hl_3 = self.pool3_h(x_sec_hl_3)
            x_sec_hh_3 = self.pool3_h(x_sec_hh_3)
            x_sec_lh_4 = self.pool4_h(x_sec_lh_4)
            x_sec_hl_4 = self.pool4_h(x_sec_hl_4)
            x_sec_hh_4 = self.pool4_h(x_sec_hh_4)

            return [[(torch.cat((x_lh_l1, x_hl_l1, x_hh_l1), dim=1),
                      torch.cat((x_sec_lh_1, x_sec_hl_1, x_sec_hh_1), dim=1)),
                     (torch.cat((x_lh_l2, x_hl_l2, x_hh_l2), dim=1),
                      torch.cat((x_sec_lh_2, x_sec_hl_2, x_sec_hh_2), dim=1)),
                     (torch.cat((x_lh_l3, x_hl_l3, x_hh_l3), dim=1),
                      torch.cat((x_sec_lh_3, x_sec_hl_3, x_sec_hh_3), dim=1)),
                     (torch.cat((x_lh_l4, x_hl_l4, x_hh_l4), dim=1),
                      torch.cat((x_sec_lh_4, x_sec_hl_4, x_sec_hh_4), dim=1))],
                    [(torch.cat((x_lh_h_1, x_hl_h_1, x_hh_h_1), dim=1),
                      torch.cat((x_sec_lh_h_1, x_sec_hl_h_1, x_sec_hh_h_1),
                                dim=1)),
                     (torch.cat((x_lh_h_2, x_hl_h_2, x_hh_h_2), dim=1),
                      torch.cat((x_sec_lh_h_2, x_sec_hl_h_2, x_sec_hh_h_2),
                                dim=1)),
                     (torch.cat((x_lh_h_3, x_hl_h_3, x_hh_h_3), dim=1),
                      torch.cat((x_sec_lh_h_3, x_sec_hl_h_3, x_sec_hh_h_3),
                                dim=1)),
                     (torch.cat((x_lh_h_4, x_hl_h_4, x_hh_h_4), dim=1),
                      torch.cat((x_sec_lh_h_4, x_sec_hl_h_4, x_sec_hh_h_4),
                                dim=1))],
                    x_out]

        else:
            return x_out


if __name__ == '__main__':
    import time

    x = torch.randn((16, 3, 16, 224, 224)).cuda()
    y = torch.randn((16, 3, 16, 224, 224)).cuda()
    t = torch.randn((1, 1))
    model = ResNet3d_pro(BasicBlock, [2, 2, 2, 2], conv_type='vanilla').cuda()
    mse = nn.MSELoss().cuda()
    while True:
        out_map_3d, out_map_time, out_cls = model(stego=x, secret=y, is_train=True)
        # start_time = time.time()
    loss_1 = mse(out_map_time[0][0], out_map_time[0][1])
    loss_2 = mse(out_map_time[1][0], out_map_time[1][1])
    loss_3 = mse(out_map_time[2][0], out_map_time[2][1])
    loss_4 = mse(out_map_time[3][0], out_map_time[3][1])
    end_time = time.time()
    print(str(end_time - start_time))
