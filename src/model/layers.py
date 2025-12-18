import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def dwt_init3d(x):
    x01 = x[:, :, :, 0::2, :] / 2
    x02 = x[:, :, :, 1::2, :] / 2
    x1 = x01[:, :, :, :, 0::2]
    x2 = x02[:, :, :, :, 0::2]
    x3 = x01[:, :, :, :, 1::2]
    x4 = x02[:, :, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_LH, x_HL, x_HH


def dwt_time(x):
    x_even = x[:, :, 0::2, :, :]
    x_odd = x[:, :, 1::2, :, :]

    x_even = x_even / 2.
    x_odd = x_odd / 2.

    x_low = x_even + x_odd
    x_high = -x_even + x_odd

    return x_low, x_high


class DWTtime(nn.Module):
    def __init__(self):
        super(DWTtime, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x_l, x_h = dwt_time(x)
        return x_l, x_h


class DWT3d(nn.Module):
    def __init__(self):
        super(DWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init3d(x)


def iwt_init3d(x_LL, x_LH, x_HL, x_HH):
    # 计算四个分量
    x1 = (x_LL - x_HL - x_LH + x_HH) / 2
    x2 = (x_LL - x_HL + x_LH - x_HH) / 2
    x3 = (x_LL + x_HL - x_LH - x_HH) / 2
    x4 = (x_LL + x_HL + x_LH + x_HH) / 2

    # 获取输入张量的形状信息
    batch, channel, depth, height, width = x_LL.shape

    # 初始化重建后的张量（空间维度扩大为2倍）
    x_recon = torch.zeros(batch, channel, depth, height * 2, width * 2,
                          device=x_LL.device, dtype=x_LL.dtype)

    # 将四个分量填充到对应位置
    x_recon[:, :, :, 0::2, 0::2] = x1  # 偶行偶列
    x_recon[:, :, :, 1::2, 0::2] = x2  # 奇行偶列
    x_recon[:, :, :, 0::2, 1::2] = x3  # 偶行列奇
    x_recon[:, :, :, 1::2, 1::2] = x4  # 奇行列奇

    return x_recon


class IWT3d(nn.Module):
    def __init__(self):
        super(IWT3d, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_LH, x_HL, x_HH):
        return iwt_init3d(x_LL, x_LH, x_HL, x_HH)


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(in_channels=inplanes, out_channels=planes, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(in_channels=planes, out_channels=planes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(in_channels=planes, out_channels=planes, stride=stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CrossTemporalAttention(nn.Module):
    def __init__(self, in_dim, dim_reduction=2, size_reduction=2, with_sa=True, max_temporal_len=1000, theta=0.0):
        super().__init__()
        self.size_reduction = size_reduction
        self.dim_reduction = dim_reduction
        self.with_sa = with_sa
        self.theta = theta

        self.down_conv = nn.Conv3d(in_dim, in_dim // dim_reduction, kernel_size=1, stride=1)
        self.down_conv_c = nn.Conv3d(in_dim, in_dim // dim_reduction, kernel_size=1, stride=1)

        self.temp_dim = in_dim // dim_reduction
        self.rotary_embed = RotaryPositionalEmbedding(self.temp_dim // self.size_reduction,
                                                      max_temporal_len // self.size_reduction)
        self.rotary_embed_c = RotaryPositionalEmbedding(self.temp_dim // self.size_reduction,
                                                        max_temporal_len // self.size_reduction)

        self.query_conv = nn.Conv3d(self.temp_dim, self.temp_dim // self.size_reduction, kernel_size=1)
        self.key_conv = nn.Conv3d(self.temp_dim, self.temp_dim // self.size_reduction, kernel_size=1)
        self.value_conv = nn.Conv3d(self.temp_dim, self.temp_dim, kernel_size=1)

        if with_sa:
            self.query_conv_sa = nn.Conv3d(self.temp_dim, self.temp_dim // self.size_reduction, kernel_size=1)
            self.key_conv_sa = nn.Conv3d(self.temp_dim, self.temp_dim // self.size_reduction, kernel_size=1)
            self.value_conv_sa = nn.Conv3d(self.temp_dim, self.temp_dim, kernel_size=1)

        self.norm = nn.GroupNorm(1, self.temp_dim)
        self.up_conv = nn.Conv3d(self.temp_dim, in_dim, kernel_size=1, stride=1)

        if with_sa:
            self.norm_sa = nn.GroupNorm(1, self.temp_dim)
            self.up_conv_sa = nn.Conv3d(self.temp_dim, in_dim, kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=-1)

    def apply_rotary_pos_emb(self, x, sin_embed, cos_embed):
        # x: [B, C, T, H, W]
        x1, x2 = x.chunk(2, dim=1)

        # 旋转操作: [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
        x_rot = torch.cat(
            (x1 * cos_embed - x2 * sin_embed,
             x2 * cos_embed + x1 * sin_embed),
            dim=1
        )
        return x_rot

    def forward(self, x, x_c):
        batch_size, C, T, H, W = x.size()
        temp_T = T // self.size_reduction
        temp_H = H // self.size_reduction
        temp_W = W // self.size_reduction

        x_down = F.interpolate(x, size=(temp_T, temp_H, temp_W), mode='trilinear', align_corners=True)
        x_down = self.down_conv(x_down)

        x_c_down = F.interpolate(x_c, size=(temp_T, temp_H, temp_W), mode='trilinear', align_corners=True)
        x_c_down = self.down_conv_c(x_c_down)

        sin_embed, cos_embed = self.rotary_embed(temp_T, temp_H, temp_W)
        sin_embed_c, cos_embed_c = self.rotary_embed_c(temp_T, temp_H, temp_W)

        if self.theta != 0.0:
            query = self.query_conv(x_down)
            query = self.apply_rotary_pos_emb(query, sin_embed, cos_embed)
            query = query.view(batch_size, -1, temp_T * temp_H * temp_W).permute(0, 2, 1)  # [B, N, C//8]

            key = self.key_conv(x_c_down)
            key = self.apply_rotary_pos_emb(key, sin_embed_c, cos_embed_c)
            key = key.view(batch_size, -1, temp_T * temp_H * temp_W)  # [B, C//8, N]

            value = self.value_conv(x_c_down).view(batch_size, -1, temp_T * temp_H * temp_W)  # [B, C, N]

            attn_scores = torch.bmm(query, key) / math.sqrt(self.temp_dim // 8)
            attn_weights = self.softmax(attn_scores)  # [B, N, N]

            out = torch.bmm(attn_weights, value.permute(0, 2, 1))  # [B, N, C]
            out = out.permute(0, 2, 1)  # [B, C, N]
            out = out.view(batch_size, self.temp_dim, temp_T, temp_H, temp_W)
            out = self.norm(out)
            out = self.up_conv(out)
            out = F.interpolate(out, size=(T, H, W), mode='trilinear', align_corners=True)

        if self.with_sa:
            query_sa = self.query_conv_sa(x_c_down)
            query_sa = self.apply_rotary_pos_emb(query_sa, sin_embed, cos_embed)
            query_sa = query_sa.view(batch_size, -1, temp_T * temp_H * temp_W).permute(0, 2, 1)

            key_sa = self.key_conv_sa(x_c_down)
            key_sa = self.apply_rotary_pos_emb(key_sa, sin_embed, cos_embed)
            key_sa = key_sa.view(batch_size, -1, temp_T * temp_H * temp_W)

            value_sa = self.value_conv_sa(x_c_down).view(batch_size, -1, temp_T * temp_H * temp_W)

            attn_scores_sa = torch.bmm(query_sa, key_sa) / math.sqrt(self.temp_dim // 8)
            attn_weights_sa = self.softmax(attn_scores_sa)
            out_sa = torch.bmm(attn_weights_sa, value_sa.permute(0, 2, 1)).permute(0, 2, 1)
            out_sa = out_sa.view(batch_size, self.temp_dim, temp_T, temp_H, temp_W)
            out_sa = self.norm_sa(out_sa)
            out_sa = self.up_conv_sa(out_sa)
            out_sa = F.interpolate(out_sa, size=(T, H, W), mode='trilinear', align_corners=True)
            
            if self.theta != 0.0:
                return out_sa + x_c - self.theta * out
            else:
                return out_sa + x_c

        return x_c - self.theta * out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_temporal_len=1000):
        super().__init__()
        self.dim = dim
        self.max_len = max_temporal_len

        # 预计算正弦/余弦位置编码
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_temporal_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        self.register_buffer('emb', emb)  # [T, D]

        self.offset = nn.Parameter(torch.zeros(1, dim, max_temporal_len, 1, 1))

    def forward(self, T, H, W):
        """
        返回:
            sin_embed: [1, D, T, 1, 1]
            cos_embed: [1, D, T, 1, 1]
        """
        emb = self.emb[:T]  # [T, D]
        emb = emb.view(1, T, self.dim)  # [1, T, D]
        emb = emb.permute(0, 2, 1)  # [1, D, T]
        emb = emb.unsqueeze(-1).unsqueeze(-1)  # [1, D, T, 1, 1]

        offset = self.offset[:, :, :T, :, :]

        # 拆分为正弦和余弦分量
        sin_embed = emb[:, :self.dim // 2]
        cos_embed = emb[:, self.dim // 2:]

        return sin_embed + offset[:, :self.dim // 2], cos_embed + offset[:, :self.dim // 2]


if __name__ == '__main__':
    x = torch.randn(1, 1, 8, 8, 8).int()
    # 正向变换
    x_LL, x_LH, x_HL, x_HH = dwt_init3d(x)
    # 逆向变换
    x_recon = iwt_init3d(x_LL, x_LH, x_HL, x_HH)
    # 验证重建效果
    print("Reconstruction error:", torch.abs(x - x_recon).max().item())
