#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

from transform import *

IMG_DIR = "reduced_complex_segmentation_img/batch_1"
MASK_DIR = "reduced_complex_segmentation_single_channel_mask/batch_1"

class FaceMask(Dataset):
    def __init__(self, rootpth="/workspace/bisenet_training/complex_seg_bisenet", cropsize=(640, 480), mode='train', *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth

        self.imgs = os.listdir(os.path.join(self.rootpth, IMG_DIR))

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        impth = self.imgs[idx]
        # print(f"impth: {impth}, impth[:-3]: {os.path.splitext(impth)[0]}")
        img = Image.open(osp.join(self.rootpth, IMG_DIR, impth))
        img = img.resize((512, 512), Image.BILINEAR)
        label = Image.open(osp.join(self.rootpth, MASK_DIR, os.path.splitext(impth)[0]+'_single_channel.png')).convert('P')
        # print(np.unique(np.array(label)))
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    # face_data = '/home/zll/data/CelebAMask-HQ/CelebA-HQ-img'
    # face_sep_mask = '/home/zll/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
    # mask_path = '/home/zll/data/CelebAMask-HQ/mask'
    # counter = 0
    # total = 0
    # for i in range(15):
    #     # files = os.listdir(osp.join(face_sep_mask, str(i)))

    #     atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #             'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    #     for j in range(i*2000, (i+1)*2000):

    #         mask = np.zeros((512, 512))

    #         for l, att in enumerate(atts, 1):
    #             total += 1
    #             file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
    #             path = osp.join(face_sep_mask, str(i), file_name)

    #             if os.path.exists(path):
    #                 counter += 1
    #                 sep_mask = np.array(Image.open(path).convert('P'))
    #                 # print(np.unique(sep_mask))

    #                 mask[sep_mask == 225] = l
    #         cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
    #         print(j)

    # print(counter, total)

    print("FaceMask dataset caller")

