import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import random
import pickle
import json, glob
import math
# import cv2
# from tqdm import tqdm
from PIL import Image
import time
import torchvision
import torchvision.transforms as trans
# from decord import VideoReader

class vispr_data(Dataset):

    def __init__(self, args, datasplit, shuffle = True, data_percentage = 1.0, vispr='1'):
        self.args = args
        self.datasplit = datasplit
        if self.datasplit == 'train':
            self.datapath = os.path.join(f'dataset/vispr/vispr{vispr}/train')
            self.all_paths = glob.glob(self.datapath + '/*.jpg')
            self.labels = pickle.load(open(os.path.join(f'dataset/vispr/vispr{vispr}', 'train.pkl'), 'rb'))
        elif self.datasplit == 'test':
            self.datapath = os.path.join(f'dataset/vispr/vispr{vispr}/test')
            self.labels = pickle.load(open(os.path.join(f'dataset/vispr/vispr{vispr}', 'test.pkl'), 'rb'))

            self.all_paths = glob.glob(self.datapath + '/*.jpg')                    
        
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.TENSOR = trans.ToTensor()
        self.erase_size = 19

    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):        
        clip, label, cover, vid_path = self.process_data(index)
        # if clip == None:
        #     pass
        return [clip], label, cover, vid_path, []

    def process_data(self, idx):
        if self.args.hide:
            if self.args.cov_ran:
                cover_dir = os.path.join(self.coco_dir, random.choice(os.listdir(self.coco_dir)))
                cover = Image.open(cover_dir).convert('RGB')
            else:
                cover = Image.open('dataset/COCO/train/000000002006.jpg')
            cover = cover.resize((self.args.reso_w, self.args.reso_h))
            cover = self.TENSOR(cover)
            cover = torch.stack([cover], dim=0)
        # label_building
        img_path = self.data[idx]
        # if '94043130' in img_path:
        #     pass
        # print(vid_path)
        # exit()
        label = self.labels[img_path.split('/')[-1]]
        
        # clip_building
        img = self.build_image(img_path)
        
        if self.args.hide:
            return img, label, cover, img_path
        else:
            
            return img, label, torch.tensor(0), img_path

    def build_image(self, img_path):

        try:
            # img = self.TENSOR(Image.open(img_path).convert("RGB"))
            img = torchvision.io.read_image(img_path)
            if img.shape[0] == 1:
                # print(img.shape)
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3, ...]
            if not img.shape[0]==3:
                print(f'{img_path} has {img.shape[0]} channels')
                return None
            # print(img.shape)
            # exit()
            ori_reso_w = img.shape[-1]
            ori_reso_h = img.shape[1]

            random_array = np.random.rand(2,8)
            x_erase = np.random.randint(0,self.args.reso_h, size = (2,))
            y_erase = np.random.randint(0,self.args.reso_w, size = (2,))


            cropping_factor1 = np.random.uniform(0.6, 1, size = (2,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = np.random.randint(0, ori_reso_w - ori_reso_w*cropping_factor1[0] + 1) 
            y0 = np.random.randint(0, ori_reso_h - ori_reso_h*cropping_factor1[0] + 1)

            contrast_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            hue_factor1 = np.random.uniform(-0.05,0.05, size = (2,))
            saturation_factor1 = np.random.uniform(0.9,1.1, size = (2,))
            brightness_factor1 = np.random.uniform(0.9,1.1,size = (2,))
            gamma1 = np.random.uniform(0.85,1.15, size = (2,))


            erase_size1 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            erase_size2 = np.random.randint(int(self.erase_size/2),self.erase_size, size = (2,))
            random_color_dropped = np.random.randint(0,3,(2))

            if self.datasplit == 'train':
                img = self.augmentation(img, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0], x0, y0, contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],                           gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0])/255.0
            elif self.datasplit == 'test':
                img = self.test_augmentation(img)/255.0
            
            try:
                assert(len(img.shape)!=0)
                
                return img
            except:
                # print(frames_full)
                print(f'Image {img_path} Failed')
                return None   

        except:
            print(f'Image {img_path} Failed')
            return None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        # ori_reso_h = image.shape[]
        ori_reso_h,ori_reso_w = image.shape[1], image.shape[-1]

        image = trans.functional.resized_crop(image,y0,x0,int(ori_reso_h*cropping_factor1),int(ori_reso_w*cropping_factor1),(self.args.reso_h,self.args.reso_w))


        if random_array[0] < 0.125/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3/2 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
        if random_array[2] < 0.3/2 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3/2 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125/2 and random_array[0] < 0.25/2:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.9:
            image = trans.functional.rgb_to_grayscale(image, num_output_channels = 3)
            if random_array[5] > 0.25:
                image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        # image = trans.functional.to_tensor(image)

        if random_array[6] < 0.5/2 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image


    def test_augmentation(self, image):
        h,w = image.shape[1], image.shape[-1]
        side = min(h,w)
        image = trans.functional.center_crop(image, side)
        image = trans.functional.resize(image,(self.args.reso_h,self.args.reso_w))
        return image


# if __name__ == '__main__':

    # train_dataset = vispr_dataset_generator(datasplit = 'train', shuffle = False, data_percentage = 1.0)
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, \
    #     shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_train)

    # print(f'Step involved: {len(train_dataset)/self.args.batch_size}')
    # t=time.time()

    # for i, (clip, label, vid_path) in enumerate(train_dataloader):
    #     if i%10 == 0:
    #         print()
    #         # clip = clip.permute(0,1,3,4,2)
    #         print(f'Full_clip shape is {clip[0]}')
    #         print(f'Label is {label}')
    #         # pickle.dump(clip, open('f_clip.pkl','wb'))
    #         # pickle.dump(label, open('label.pkl','wb'))
    #         # exit()
    # print(f'Time taken to load data is {time.time()-t}')

    # train_dataset = multi_baseline_dataloader_val_strong(shuffle = False, data_percentage = 1.0,  mode = 4)
    # train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=collate_fn2)

    # print(f'Step involved: {len(train_dataset)/self.args.batch_size}')
    # t=time.time()

    # for i, (clip, label, vid_path, _) in enumerate(train_dataloader):
    #     if i%25 == 0:
    #         print()
    #         # clip = clip.permute(0,1,3,4,2)
    #         print(f'Full_clip shape is {clip.shape}')
    #         print(f'Label is {label}')
    #         # print(f'Frame list is {frame_list}')
            
    #         # pickle.dump(clip, open('f_clip.pkl','wb'))
    #         # pickle.dump(label, open('label.pkl','wb'))
    #         # exit()
    # print(f'Time taken to load data is {time.time()-t}')