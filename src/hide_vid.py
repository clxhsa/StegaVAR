import time

import torch
from torch import nn
import torch.nn.functional as F

from hide.LF_VSN.models.modules import common
from hide.LF_VSN.models.networks import define_G_v2
import hide.LF_VSN.options.options as option
from src.model.get_model import load_network
from hide.HiNet.model import Hinet
from hide.wengnet.wengnet import wengnet
from hide.hidden.hidden import HiDDeN

def hide_LF(img_batch, cover_batch, hidden_net, dwt, iwt):
    img_batch = img_batch.unsqueeze(1)
    forw_L = []
    gop = 3
    b, n, t, c, h, w = img_batch.shape
    ref_L = torch.cat([img_batch[:, :, 0:1, :, :, :], img_batch, img_batch[:, :, t - 1:t, :, :, :]], dim=2)
    real_H = torch.cat([cover_batch[:, 0:1, :, :, :], cover_batch, cover_batch[:, t - 1:t, :, :, :]], dim=1)
    id = 0
    for j in range(t):
        # forward downscaling
        host = real_H[:, j:j + 3]
        secret = ref_L[:, :, j:j + 3]
        secret = [dwt(secret[:, i].reshape(b, -1, h, w)) for i in range(n)]
        output, out_h = hidden_net(x=dwt(host.reshape(b, -1, h, w)), x_h=secret)
        output = iwt(output)
        out_lrs = output[:, :3 * gop, :, :].reshape(-1, gop, 3, h, w)
        forw_L.append(out_lrs[:, gop // 2])

    return torch.clamp(torch.stack(forw_L, dim=1), 0, 1).squeeze(1)


def hide_hi(img_batch, cover_batch, hidden_net, dwt, iwt):
    # print(img_batch.shape)
    b, t, c, h, w = img_batch.shape
    stego_L = []
    for j in range(t):
        cover_input = dwt(cover_batch[:, j:j + 1].squeeze(1))
        secret_input = dwt(img_batch[:, j:j + 1].squeeze(1))
        # print(secret_input.shape)
        input_img = torch.cat((cover_input, secret_input), 1)
        output = hidden_net(input_img)
        output_stego = output.narrow(1, 0, 4 * c)
        stego_img = iwt(output_stego)
        # print(stego_img.shape)
        stego_L.append(stego_img)
        # output = torch.clamp(torch.stack(stego_L, dim=1), 0, 1)
        # print(output.shape)

    return torch.clamp(torch.stack(stego_L, dim=1), 0, 1)


def hide_weng(img_batch, cover_batch, hidden_net):
    b, t, c, h, w = img_batch.shape
    pad_size = (256 - h) // 2
    stego_L = []
    for j in range(t):
        cover_input = cover_batch[:, j:j + 1].squeeze(1)
        secret_input = img_batch[:, j:j + 1].squeeze(1)
        padded_cover = F.pad(cover_input, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        padded_secret = F.pad(secret_input, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
        # print(secret_input.shape)
        stego_img = hidden_net(padded_secret, padded_cover)
        stego_img = stego_img[:, :, pad_size:-pad_size, pad_size:-pad_size]
        stego_L.append(stego_img)

    return torch.clamp(torch.stack(stego_L, dim=1), 0, 1)


def hide_hidden(img_batch, cover_batch, hidden_net):
    b, t, c, h, w = img_batch.shape
    stego_L = []
    for j in range(t):
        cover_input = cover_batch[:, j:j + 1].squeeze(1)
        secret_input = img_batch[:, j:j + 1].squeeze(1)
        stego_img = hidden_net(cover_input, secret_input)
        # print(stego_img.shape)
        stego_L.append(stego_img)
        # output = torch.clamp(torch.stack(stego_L, dim=1), 0, 1)
        # print(output.shape)

    return torch.clamp(torch.stack(stego_L, dim=1), 0, 1)


def hide_vid(args, img_batch, cover_batch, hidden_net, dwt, iwt):
    # print(img_batch.shape)
    # if args.task == 'pri':
    #     img_batch = img_batch.unsqueeze(2)
    #     cover_batch = cover_batch.unsqueeze(2)
    
    if args.hide_model == 'lfvsn':
        output = hide_LF(img_batch, cover_batch, hidden_net, dwt, iwt)
    elif args.hide_model == 'hinet':
        output = hide_hi(img_batch, cover_batch, hidden_net, dwt, iwt)
    elif args.hide_model == 'wengnet':
        output = hide_weng(img_batch, cover_batch, hidden_net)
    elif args.hide_model == 'hidden':
        output = hide_hidden(img_batch, cover_batch, hidden_net)

    return output
