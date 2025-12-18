import json
import numpy as np
import os
import random
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
import traceback
import logging
import cv2
from PIL import Image
import pickle

import sys

sys.path.insert(0, '..')


# Training dataloader.
class hmdb_train(Dataset):

    def __init__(self, args, shuffle = True, data_percentage=1.0, split=1):
        self.args = args
        if split == 1:
            self.all_paths = open('dataset/VAR/hmdb51/train.txt',
                                  'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')

        if args.task == 'har':
            self.classes = json.load(open('dataset/VAR/hmdb51/action_classes.json'))['classes']
        elif args.task == 'pri':
            self.classes = pickle.load(open('dataset/VAR/hmdb51/hmdb51_privacy_attribute_label.pickle', 'rb'))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths) * self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

        self.ori_reso_h = 240
        self.ori_reso_w = 320

        self.coco_dir = 'dataset/COCO/train'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        clip, label, cover, vid_path = self.process_data(index)
        return clip, label, cover, vid_path

    def process_data(self, idx):
        if self.args.hide:
            # cover_dir = os.path.join(self.coco_dir, random.choice(os.listdir(self.coco_dir)))
            # cover = Image.open(cover_dir).convert('RGB')
            cover = Image.open('dataset/COCO/train/000000002006.jpg')
            cover = cover.resize((self.args.reso_w, self.args.reso_h))
            cover = self.TENSOR(cover)
            cover = torch.stack([cover for i in range(self.args.num_frames)], dim=0)

        # label_building
        #         vid_path = self.data[idx]
        vid_path = 'dataset/VAR/hmdb51' + '/hmdb51/' + \
                   self.data[idx].split(' ')[0][:-4]
        # print(vid_path)
        # exit()
        if self.args.task == 'har':
            label = self.classes[vid_path.split('/')[-2]] - 1  # make label start from zero
        elif self.args.task == 'pri':
            label = self.classes[self.data[idx]]
            label = np.squeeze(label)
        #         label = self.classes[vid_path.split('/')[6]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS

        # clip_building
        if self.args.task == 'har':
            clip = self.build_clip(vid_path)
        elif self.args.task == 'pri':
            # 获取文件夹中的所有jpg文件并按数字排序
            img_files = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            img_files.sort(key=lambda x: int(x.split('.')[0]))
            clip = random.choice(img_files)
            # clip = self.augmentation(cv2.imread(clip))
            clip = Image.open(os.path.join(vid_path, clip))
            clip = clip.resize((self.args.reso_w, self.args.reso_h))
            clip = [self.TENSOR(clip)]

        if self.args.hide:
            return clip, label, cover, vid_path
        else:
            return clip, label, torch.tensor(0), vid_path

    def build_clip(self, vid_path):
        try:
            # 获取文件夹中的所有jpg文件并按数字排序
            img_files = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            img_files.sort(key=lambda x: int(x.split('.')[0]))
            frame_count = len(img_files)
            if frame_count == 0:
                print(f"Clip {vid_path} has no frames")
                return None

            # 生成帧索引列表的逻辑保持不变
            skip_frames_full = self.args.fix_skip
            left_over = frame_count - self.args.fix_skip * self.args.num_frames
            if left_over <= 0:
                start_frame_full = 0
            else:
                start_frame_full = np.random.randint(0, int(left_over))

            frames_full = start_frame_full + np.array(
                [int(skip_frames_full * f) for f in range(self.args.num_frames)])
            frames_full = frames_full.astype(int)
            if frames_full[-1] >= frame_count:
                frames_full[-1] = frame_count - 1

            # 生成随机增强参数（与原代码完全一致）
            random_array = np.random.rand(2, 8)
            x_erase = np.random.randint(0, self.args.reso_h, size=(2,))
            y_erase = np.random.randint(0, self.args.reso_w, size=(2,))
            cropping_factor1 = np.random.uniform(0.6, 1, size=(2,))
            x0 = np.random.randint(0, self.ori_reso_w - int(self.ori_reso_w * cropping_factor1[0]) + 1)
            y0 = np.random.randint(0, self.ori_reso_h - int(self.ori_reso_h * cropping_factor1[0]) + 1)
            contrast_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            hue_factor1 = np.random.uniform(-0.05, 0.05, size=(2,))
            saturation_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            brightness_factor1 = np.random.uniform(0.9, 1.1, size=(2,))
            gamma1 = np.random.uniform(0.85, 1.15, size=(2,))
            erase_size1 = np.random.randint(int(self.erase_size / 2), self.erase_size, size=(2,))
            erase_size2 = np.random.randint(int(self.erase_size / 2), self.erase_size, size=(2,))
            random_color_dropped = np.random.randint(0, 3, (2,))

            # 读取并处理目标帧
            full_clip = []
            for frame_num in frames_full:
                if frame_num >= frame_count:
                    frame_num = frame_count - 1

                img_path = os.path.join(vid_path, img_files[frame_num])
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Failed to load frame {frame_num} in {vid_path}")
                    return None

                if self.args.weak_aug:
                    augmented_frame = self.weak_augmentation(frame, cropping_factor1[0], x0, y0)
                else:
                    augmented_frame = self.augmentation(
                        frame,
                        random_array[0],
                        x_erase[0],
                        y_erase[0],
                        cropping_factor1[0],
                        x0, y0,
                        contrast_factor1[0],
                        hue_factor1[0],
                        saturation_factor1[0],
                        brightness_factor1[0],
                        gamma1[0],
                        erase_size1[0],
                        erase_size2[0],
                        random_color_dropped[0]
                    )
                full_clip.append(augmented_frame)

            # 处理帧数不足的情况（与原代码完全一致）
            if len(full_clip) < self.args.num_frames and len(full_clip) > (self.args.num_frames / 2):
                remaining_num_frames = self.args.num_frames - len(full_clip)
                full_clip += full_clip[::-1][1:remaining_num_frames + 1]

            try:
                assert len(full_clip) == self.args.num_frames
                return full_clip
            except AssertionError:
                print(f'Clip {vid_path} Failed')
                return None

        except Exception as e:
            print(f'Clip {vid_path} Failed: {str(e)}')
            return None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1, \
                     x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1, \
                     gamma1, erase_size1, erase_size2, random_color_dropped):

        image = self.PIL(image)
        image = trans.functional.resized_crop(image, y0, x0, int(self.ori_reso_h * cropping_factor1),
                                              int(self.ori_reso_h * cropping_factor1),
                                              (self.args.reso_h, self.args.reso_w))

        if random_array[0] < 0.125 / 2:
            image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1)  # 0.75 to 1.25
        if random_array[1] < 0.3 / 2:
            image = trans.functional.adjust_hue(image,
                                                hue_factor=hue_factor1)  # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
        if random_array[2] < 0.3 / 2:
            image = trans.functional.adjust_saturation(image,
                                                       saturation_factor=saturation_factor1)  # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3 / 2:
            image = trans.functional.adjust_brightness(image,
                                                       brightness_factor=brightness_factor1)  # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125 / 2 and random_array[0] < 0.25 / 2:
            image = trans.functional.adjust_contrast(image, contrast_factor=contrast_factor1)  # 0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.to_grayscale(image, num_output_channels=3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma=gamma1, gain=1)  # gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5 / 2:
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0)

        return image

    def weak_augmentation(self, image, cropping_factor1, x0, y0):

        image = self.PIL(image)
        image = trans.functional.resized_crop(image, y0, x0, int(self.ori_reso_h * cropping_factor1),
                                              int(self.ori_reso_w * cropping_factor1),
                                              (self.args.reso_h, self.args.reso_w), antialias=True)

        image = trans.functional.to_tensor(image)

        return image


# Validation dataset.
class hmdb_val(Dataset):

    def __init__(self, args, shuffle = True, data_percentage=1.0, mode=0, skip=1, \
                 hflip=0, cropping_factor=1.0, split=1):
        self.args = args

        if split == 1:
            self.all_paths = open('dataset/VAR/hmdb51/test.txt',
                                  'r').read().splitlines()
        else:
            print(f'Invalid split input: {split}')

        if args.task == 'har':
            self.classes = json.load(open('dataset/VAR/hmdb51/action_classes.json'))['classes']
        elif args.task == 'pri':
            self.classes = pickle.load(open('dataset/VAR/hmdb51/hmdb51_privacy_attribute_label.pickle', 'rb'))
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)

        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths) * self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()
        self.mode = mode
        self.skip = skip
        self.hflip = hflip
        self.cropping_factor = cropping_factor

        self.ori_reso_h = 240
        self.ori_reso_w = 320
        
        self.coco_dir = 'dataset/COCO/train'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clip, label, cover, vid_path = self.process_data(index)
        return clip, label, cover, vid_path, []

    def process_data(self, idx):
        if self.args.hide:
            # cover_dir = os.path.join(self.coco_dir, random.choice(os.listdir(self.coco_dir)))
            # cover = Image.open(cover_dir).convert('RGB')
            cover = Image.open('dataset/COCO/train/000000002006.jpg')
            cover = cover.resize((self.args.reso_w, self.args.reso_h))
            cover = self.TENSOR(cover)
            cover = torch.stack([cover for i in range(self.args.num_frames)], dim=0)

        # label_building
        vid_path = 'dataset/VAR/hmdb51' + '/hmdb51/' + \
                   self.data[idx].split(' ')[0][:-4]
        # print(vid_path)
        # exit()
        if self.args.task == 'har':
            label = self.classes[vid_path.split('/')[-2]] - 1  # make label start from zero
        elif self.args.task == 'pri':
            label = self.classes[self.data[idx]]
            label = np.squeeze(label)
        #         label = self.classes[vid_path.split('/')[6]] # THIS MIGHT BE DIFFERNT AFTER STEVE MOVE THE PATHS

        # clip_building
        if self.args.task == 'har':
            clip = self.build_clip(vid_path)
        elif self.args.task == 'pri':
            # 获取文件夹中的所有jpg文件并按数字排序
            img_files = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            img_files.sort(key=lambda x: int(x.split('.')[0]))
            clip = random.choice(img_files)
            # clip = self.augmentation(cv2.imread(clip))
            clip = Image.open(os.path.join(vid_path, clip))
            clip = clip.resize((self.args.reso_w, self.args.reso_h))
            clip = [self.TENSOR(clip)]

        if self.args.hide:
            return clip, label, cover, vid_path
        else:
            return clip, label, torch.tensor(0), vid_path

    def build_clip(self, vid_path):
        try:
            # 获取文件夹中的所有jpg文件并按数字排序
            img_files = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            img_files.sort(key=lambda x: int(x.split('.')[0]))
            frame_count = len(img_files)
            if frame_count == 0:
                print(f"Clip {vid_path} has no frames")
                return None

            # 生成帧索引列表的逻辑保持不变
            skip_frames_full = self.args.fix_skip
            # left_over = frame_count - self.args.fix_skip * self.args.num_frames

            # if left_over <= 0:
            #     start_frame_full = 0
            # else:
            #     start_frame_full = np.random.randint(0, int(left_over))

            left_over = skip_frames_full * self.args.num_frames

            start_frame_full = 0 + int(np.linspace(0, frame_count - left_over - 10, self.args.num_modes)[self.mode])

            frames_full = start_frame_full + np.array(
                [int(skip_frames_full * f) for f in range(self.args.num_frames)])
            frames_full = frames_full.astype(int)
            if frames_full[-1] >= frame_count:
                frames_full[-1] = frame_count - 1

            full_clip = []
            for frame_num in frames_full:
                if frame_num >= frame_count:
                    frame_num = frame_count - 1

                img_path = os.path.join(vid_path, img_files[frame_num])
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Failed to load frame {frame_num} in {vid_path}")
                    return None

                augmented_frame = self.augmentation(frame)
                full_clip.append(augmented_frame)

            if len(full_clip) < self.args.num_frames and len(full_clip) > (self.args.num_frames / 2):
                remaining_num_frames = self.args.num_frames - len(full_clip)
                full_clip += full_clip[::-1][1:remaining_num_frames + 1]

            try:
                assert len(full_clip) == self.args.num_frames
                return full_clip
            except AssertionError:
                print(f'Clip {vid_path} Failed')
                return None

        except Exception as e:
            print(f'Clip {vid_path} Failed: {str(e)}')
            return None

    def augmentation(self, image):
        image = self.PIL(image)
        # image = trans.functional.resize(image, (self.args.reso_h, self.args.reso_w), antialias=True)

        if self.cropping_factor <= 1:
            image = trans.functional.center_crop(image, (
                int(self.ori_reso_h * self.cropping_factor), int(self.ori_reso_h * self.cropping_factor)))
        image = trans.functional.resize(image, (self.args.reso_h, self.args.reso_w))
        if self.hflip != 0:
            image = trans.functional.hflip(image)

        return trans.functional.to_tensor(image)


# if __name__ == '__main__':
#     import args_anonymization as args

#     train_dataset = ucf_val(args=args, shuffle=True, data_percentage=0.1)
#     # train_dataset = ucf_val(args=args, shuffle=False, data_percentage=0.1)
#     # train_dataset = contrastive_train_dataloader(args=args, shuffle=False, data_percentage=0.1)
#     # train_dataset = contrastive_val_dataloader(args=args, shuffle=False, data_percentage=0.1)

#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
#                                   collate_fn=collate_fn_train, num_workers=0)  # args.num_workers)

#     print(f'Length of training dataset: {len(train_dataset)}')
#     print(f'Steps involved: {len(train_dataset) / args.batch_size}')
#     t = time.time()

#     for i, (clip, label, vid_path, frame_list) in enumerate(train_dataloader):
#         if i % 10 == 0:
#             print()
#             print(f'Full_clip shape is {clip.shape}')
#             # print(f'Label is {label}')
#             # print(f'Frame list is {frame_list}')
#             continue
