from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import math
import numpy as np
import torch
import cv2

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img
    
class letterbox_image_xy(object):
    def __init__(self, resize_h, resize_w):
        self.resize_h = resize_h
        self.resize_w = resize_w

    def __call__(self, image):
        new_w, new_h = image.size
        image = np.array(image)
        alpha = new_h / new_w

        if self.resize_h == 320 and self.resize_w == 128:
            if alpha < 2.0:
                new_h = int(new_h * 1.2)
        elif self.resize_h == 288 and self.resize_w == 128:
            if alpha < 1.875:
                new_h = int(new_h * 1.2)

        image = cv2.resize(image, (new_w, new_h), cv2.INTER_LANCZOS4)

        if float(self.resize_w / image.shape[1]) < float(self.resize_h / image.shape[0]):
            new_w = int(self.resize_w)
            new_h = int((image.shape[0] * self.resize_w) / image.shape[1])
        else:
            new_h = int(self.resize_h)
            new_w = int((image.shape[1] * self.resize_h) / image.shape[0])

        img_resize = cv2.resize(image, (new_w, new_h), cv2.INTER_LANCZOS4)
        box = 127 * np.ones((self.resize_h, self.resize_w, img_resize.shape[2]))

        box[int((self.resize_h -new_h) / 2):int((self.resize_h +new_h) / 2) + new_h,
            int((self.resize_w - new_w )/ 2):int((self.resize_w + new_w) / 2) + new_w, :] = img_resize[0:new_h, 0:new_w, :]
        
        #转灰度
        # gray = cv2.cvtColor(np.uint8(box), cv2.COLOR_BGR2GRAY)
        # im = np.zeros_like(box)
        # im[:,:,0] = gray
        # im[:,:,1] = gray
        # im[:,:,2] = gray
        # img = Image.fromarray(np.uint8(im))

        img = Image.fromarray(np.uint8(box))

        return img

def imgBrighrness(img, a, b):
    gamma = random.uniform(a, b)
    imgenhancer_Brighrness = ImageEnhance.Brightness(img)
    img_enhance_Brighrness = imgenhancer_Brighrness.enhance(gamma)
    return img_enhance_Brighrness

class Random_dark(object):
    def __init__(self, probability=0.5):
        
        self.probability = probability

    def __call__(self, img):
        num = random.uniform(0,1)
        if num > self.probability:
            return img
        img = imgBrighrness(img, 0.5, 0.7)
        return img
        