import torch
import numpy as np

from src.dataloader.ucf101 import ucf_train, ucf_val
from src.dataloader.hmdb51 import hmdb_train, hmdb_val
from src.dataloader.vispr import vispr_data

def get_train_data(args, data_percentage, shuffle=True):
    if args.train_data == 'ucf101':
        train_data = ucf_train(args=args, shuffle=shuffle, data_percentage=data_percentage)

    elif args.train_data == 'hmdb51':
        train_data = hmdb_train(args=args, shuffle=shuffle, data_percentage=data_percentage)
        
    elif args.train_data == 'vispr1':
        train_data = vispr_data(args=args, datasplit='train', shuffle=shuffle, data_percentage=data_percentage, vispr='1')
        
    elif args.train_data == 'vispr2':
        train_data = vispr_data(args=args, datasplit='train', shuffle=shuffle, data_percentage=data_percentage, vispr='2')

    else:
        print('No such dataset!')

    return train_data


def get_val_data(args, mode=0, cropping_factor=1.0, shuffle=True, data_percentage=1.0):
    if args.val_data == 'ucf101':
        val_data = ucf_val(args=args, shuffle=shuffle, data_percentage=data_percentage, mode=mode, cropping_factor=cropping_factor)

    elif args.val_data == 'hmdb51':
        val_data = hmdb_val(args=args, shuffle=shuffle, data_percentage=data_percentage, mode=mode, cropping_factor=cropping_factor)

    elif args.val_data == 'vispr1':
        val_data = vispr_data(args=args, datasplit='test', shuffle=shuffle, data_percentage=data_percentage, vispr='1')
        
    elif args.val_data == 'vispr2':
        val_data = vispr_data(args=args, datasplit='test', shuffle=shuffle, data_percentage=data_percentage, vispr='2')
        
    else:
        print('No such dataset!')

    return val_data


def collate_val_pri(batch):
    f_clip, label, cover, vid_path, frame_list = [], [], [], [], []

    for item in batch:
        if not (item[0] == None or item[1].any() == None or item[2] == None):
            f_clip.append(torch.stack(item[0], dim=0))
            label.append(item[1])
            cover.append(item[2])
            vid_path.append(item[3])
            frame_list.append(item[4])

    # if len(f_clip) < 2:
    #     return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(np.array(label))
    frame_list = torch.tensor(frame_list)
    cover = torch.stack(cover, dim=0)

    return f_clip, label, cover, vid_path, frame_list


def collate_train_pri(batch):
    f_clip, label, cover, vid_path = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1].any() == None or item[2] == None):
            f_clip.append(torch.stack(item[0], dim=0))
            label.append(item[1])
            cover.append(item[2])
            vid_path.append(item[3])
            # frame_list.append(item[3])

    # if len(f_clip) < 2:
    #     print(vid_path)
    #     return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(np.array(label))
    cover = torch.stack(cover, dim=0)
    # frame_list = torch.stack(frame_list, dim=0)

    return f_clip, label, cover, vid_path


def collate_val(batch):
    f_clip, label, cover, vid_path, frame_list = [], [], [], [], []

    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0], dim=0))
            label.append(item[1])
            cover.append(item[2])
            vid_path.append(item[3])
            frame_list.append(item[4])

    if len(f_clip) < 2:
        return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(label)
    frame_list = torch.tensor(frame_list)
    cover = torch.stack(cover, dim=0)

    return f_clip, label, cover, vid_path, frame_list


def collate_train(batch):
    f_clip, label, cover, vid_path = [], [], [], []
    for item in batch:
        if not (item[0] == None or item[1] == None or item[2] == None):
            f_clip.append(torch.stack(item[0], dim=0))
            label.append(item[1])
            cover.append(item[2])
            vid_path.append(item[3])
            # frame_list.append(item[3])

    if len(f_clip) < 2:
        print(vid_path)
        return None, None, None, None
    f_clip = torch.stack(f_clip, dim=0)
    label = torch.tensor(label)
    cover = torch.stack(cover, dim=0)
    # frame_list = torch.stack(frame_list, dim=0)

    return f_clip, label, cover, vid_path