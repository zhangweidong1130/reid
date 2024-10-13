from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
from utils.random_erasing import *
from PIL import Image
import collections
import random
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import math
import pickle

class train_person_kd(dataset.Dataset):
    def __init__(self, args, transform, transform_kd, dtype):

        self.transform = transform
        self.transform_kd = transform_kd
        self.loader = default_loader

        #数据增强部分
        if 1:
            self.Random_dark = Random_dark(probability=0.5)


        data_path = args.datadir
        if dtype == 'train':
            data_path += '/bounding_box_train'
        elif dtype == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        
        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        self.id_cam_index = self.creat_list(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img_name = path.split('/')
        target = self._id2label[self.id(path)]
        img_ps = Image.open('./person.png')

        img = self.loader(path)
        if self.transform is not None:
            img_t = self.transform(img)
            img_kd = self.transform_kd(img)
            img_t_gray = self.transform_kd(img)

        return img, target, img_ps, img_name[-1], img_kd, img_t_gray

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]
    
    def creat_list(self, imgs):
        id_cam_index = collections.defaultdict(dict)
        for idx, path in enumerate(imgs):
            id_ = self.id(path)
            cam_ = self.camera(path)
            if cam_ in id_cam_index[id_]:
                id_cam_index[id_][cam_].append(idx)
            else:
                id_cam_index[id_][cam_] = [idx]
        return id_cam_index