import math
import numpy as np
import PIL.Image as Image
import torch.nn as nn
import torch
import cv2
import pickle
import shutil

def eval_faiss_zwd(query_ids, gallery_ids, query_cameras, gallery_cameras, query_imgs, gallery_imgs, indices, m, distmat_sort, save_erro, topk=200):#有道云记录
    pass