import os
import torch
import numpy as np
import utils.utility as utility
from utils.utility import printd, recursive_to_device, data_prefecher, Save_Testerro
from utils.numba_metric import eval_faiss_zwd
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking

import math
import sys
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import pickle
import sklearn import decomposition
import PIL.Image as Image
import cv2
from loss.myloss import CenterLoss, regular_weight
from numba import cuda as num_cuda


from model import zwdvit_hjw
try:
    import apex
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    fp16=False
    amp.register_float_function(torch, 'sigmoid')
    amp.register_float_function(torch, 'softmax')
    amp.register_float_function(torch, 'binary_cross_entropy_with_logits')
except:
    print('this is not an error, if you want to use lower precision, i.e., fp16')
    fp16=True
    pass
use_att=True
fp16=True
try:
    from torch.cuda.amp import autocast, GradScaler
except:
    print('please use the pytorch version >= 1.6')
    pass

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        torch.cuda.set_device(args.local_rank)
        self.device = torch.device(f'cuda:{args.local_rank}' )#if torch.cuda.is_available() else 'cpu')
        self.loader = loader
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset
        self.trainset = loader.trainset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.

        module = self.model.model

        print('init teacher model 1........')
        self.teacher_model_1 = zwdvit_hjw.MGNRESNET(args)
        techer_ckpt_1 = torch.load('./experiment/0/0zwd_hjw/model_best.pt',map_location=torch.device('cpu'))
        self.teacher_model_1.load_state_dict(techer_ckpt_1, strict=True)
        self.teacher_model_1 = self.teacher_model_1.to(self.device)
        for param in self.teacher_model_1.parameters():
            param.requires_grad = False
        self.teacher_model_1.eval()

        print('teacher model 1 init done........')

        self.train_list = None
        self.optimizer = utility.make_optimizer_vit(args, self.model, sekf.train_list)
        #self.optimizer = utility.make_optimizer(args, self.model)
        #self.scheduler = utility.make_scheduler(args, self.optimizer)
        #self.device = torch.device('cpu' if args.cpu else 'cuda')

        self.scaler = GradScaler(enabled=fp16)
        self.model.model = nn.parallel.DistributedDataParallel(self.model.model, device_ids=[self.args.local_rank], 
                    output_device=self.args.local_rank, find_unused_parameters=False)
        self.model.model = self.model.model.to(self.device)

        cudnn.benchmark = True

        if args.adjust_lr != 0:
            for g in self.optimizer.param_groups:
                if g['lr'] == 0:
                    continue        
                else:
                    g['lr'] = args.adjust_lr

        #--------------学习率变化设置----------------
        self.lr_schedule = cosine_scheduler(    
            0.0002134 * (150 *8) /256.,#arg.lr*(args.batch_size_per_gpu*8)/256.
            1e-5, #args.min_lr,
            args.epochs,
            len(self.train_loader),
            warmup_epochs=10)
        self.wd_schedule = cosine_scheduler(
            0.04,#args.weight_decay
            0.4,#args.weight_decay_end,
            args.epochs,
            len(self.train_loader))



    def train(self):

        self.loss.step()

        epoch = len(self.loss.log) + 1
        self.loader.sampler.set_epoch(epoch)
        fc_prec_meter = torch.zeros(1)

        utility.adjust_lr_vit(self.optimizer, epoch, self.args.epochs, start_decay_at_ep = 320)


        if epoch > 70 or True:
            self.trainset.imgforce = True
        if epoch > 110 or True:
            self.trainset.imgforce_RGB = False

        lr = self.optimizer.param_groups[0]['lr']
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()

        self.model.train()
        # self.model.eval()
        # self.model.model.module.layer4.train()
        # self.model.model.module.global_reduction1_2.train()
        # self.model.model.module.global_reduction2_2.train()

        g_prec_meter = utility.AverageMeter()
        self.ckpt.write_log('lr----{}'.format(lr))
        epoch_loss=[]
        epoch_loss_parts=[]
        self.imglist=[]


        for batch, (inputs, labels) in enumerate(self.train_loader):
            iteration += 1

            #---------调整学习率-------------
            # it = len(self.train_loader) * (epoch - 1) + iteration -1
            # for i, param_group in enumerate(self.optimizer.param_groups):
            #     param_group['lr'] = self.lr_schedule[it]
            #     if i == 0:
            #         param_group['weight_decay'] = self.wd_schedule[it]

            starttime = time.time()
            batches = recursive_to_device(batches, self.device)

            reid_batch = batches

            inputs, labels, ps_label, imgnames, inputs_kd, inoputs_gray = reid_batch

            inputs_reid = [inputs, 'reid', labels]
            inputs_reid_kd = [inputs_kd, 'reid', labels]

            with autocast(enabled=fp16):
                with torch.no_grad(), autocast(enabled=fp16):
                    teacher_outputs1 = self.teacher_model_1(inputs_reid)
                outputs = self.model(inputs_reid)
                loss, loss_part, g_prec = self.loss(outputs, labels, epoch, ps_label, teacher_outputs= teacher_outputs1)#  loss部分需同步

                self.scaler.scale(loss).backward()
                if iteration % 4 == 0:
                    self.scaler.step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    
            epoch_loss.append(loss)
            fc_pre = torch.tensor([labels[labels==torch.argmax(opt_lg, dim=1)].size(0)/labels.size(0) for opt_lg in outputs['fc_logits']])
            fc_pre = 0.0
            fc_pre_meter+=fc_pre

            epoch_loss_parts.append(loss_part)
            
            g_prec_meter.updata(g_prec)
            endtime = time.time()

          
            if self.args.local_rank == 0:
                self.ckpt.write_log('\r[INFO] [{}/{}] {}/{} {} {} time = {:.4f} s'.format(
                    epoch, self.args.epochs,
                    iteration, len(self.train_loader),
                    self.loss.display_loss(iteration), epoch_loss_parts[-1], endtime - starttime), 
                    end='' if iteration != len(self.train_loader) else '\n')


        if self.args.local_rank == 0:
            epo_loss = sum(epoch_loss) / len(epoch_loss)
            epoch_loss_part = sum(epoch_loss_parts) / len(epoch_loss_parts)
            
            self.ckpt.write_log('[INFO] epoch_loss_part: {}'.format(epoch_loss_part))
            self.ckpt.write_log('[INFO] fc_pre: {}'.format(fc_prec_meter/(iteration+1)))

            self.loss.end_log(len(self.train_loader))
            self.ckpt.save(self, epoch, train_save=True)
    
    def test_mp(self):
        if self.args.test_only or self.test_all:
            epoch = 0
        else:
            epoch = len(self.loss.log)
        self.ckpt.write_log('\n[INFO] Test: ' + self.args.data_test)
        self.model.eval()
        self.ckpt.add_log(torch.zeros(1,5))

        qf = self.extract_feature_mp(self.query_loader)
        gf = self.extract_feature_mp(self.test_loader)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        query_lenth = len(self.queryset.ids)
        gallery_lenth = len(self.testset.ids)
        
        query_num_samples = int(math.ceil(query_lenth * 1.0 / world_size))
        gallery_num_samples = int(math.ceil(gallery_lenth * 1.0 / world_size))

        query_indices = [query_num_samples for i in range(world_size-1)]
        query_indices.append(query_lenth - sum(query_indices))
        gallery_indices = [gallery_num_samples for i in range(world_size-1)]
        gallery_indices.append(gallery_lenth - sum(gallery_indices))

        qf_tensor=[]
        for i, num in enumerate(query_indices):
            qf_tensor.append(torch.zeros([num, qf.size(1)], dtype=qf.dtype, device=self.device))
        gf_tensor=[]
        for i, num in enumerate(gallery_indices):
            gf_tensor.append(torch.zeros([num, gf.size(1)], dtype=gf.dtype, device=self.device))

        assert qf_tensor[rank].size() == qf.size()
        assert gf_tensor[rank].size() == gf.size()
        qf_tensor[rank] = qf
        gf_tensor[rank] = gf

        for i in range(world_size):
            dist.broadcast(qf_tensor[i], src=i, async_op=False)
        printd('query broadcast done')

        for i in range(world_size):
            dist.broadcast(gf_tensor[i], src=i, async_op=False)
        printd('gallery broadcast done')

        if rank != 0:
            del qf
            del gf
            del qf_tensor
            del gf_tensor
            torch.cuda.empty_cache()
            return

        qf = torch.cat(qf_tensor, 0).cpu().numpy()
        #gf = torch.cat(gf_tensor, 0).cpu().numpy()
        tempf = torch.FloatTensor()
        for ff in gf_tensor:
            tempf = torch.cat((tempf, ff.cpu()), 0)
        gf = tempf.numpy()

        if 0:
            mean, A = self.pca(qf, 256)
            qf = np.matmul(qf - mean, A)
            print(qf.shape)
            #mean, A = self.pca(gf, 256)
            gf = np.matmul(gf - mean, A)
            print(gf.shape)

        s_time = time.time()
        print('start cdist')

        if 0:
            cosine_dist = cdist(qf, gf, metric='cosine')
        else:
            import faiss 
            qf = qf /np.linalg.norm(qf, axis=1).reshape([-1,1])
            gf = gf /np.linalg.norm(gf, axis=1).reshape([-1,1])
            index = faiss.IndexFlatIP(qf.shape[1])
            index.add(gf)
            D, I = index.search(qf, 200)
            print(D.shape, I.shape)
            #二次搜索
            # twice = I[:,0]
            # gf_new = gf[twice]
            # qf_new = (qf + gf_new) / 2
            # D, I = index.search(qf_new, 200)

            m_time = time.time()

            gallery_ids = np.array(self.testset.ids)
            gallery_cameras = np.array(self.testset.cameras)
            gallery_imgs = self.testset.imgs
            query_ids = np.array(self.queryset.ids)
            query_cameras = np.array(self.queryset.cameras)
            query_imgs = self.queryset.imgs
            m = len(query_ids)

            save_erro = 0
            m_ap, r, erro_dict = eval_faiss_zwd(query_ids, gallery_ids, query_cameras, gallery_cameras, query_imgs, gallery_imgs, I, m, D, save_erro)
            e_time = time.time()
            #---------save_imgs----------
            if save_erro:
                Save_Testerro(self.args.data_test.lower(), erro_dict)

        print('cdist_time: {:.4f}s'.format(e_time - m_time))

        self.ckpt.log[-1,0] = m_ap
        self.ckpt.log[-1,1] = r[0]
        self.ckpt.log[-1,2] = r[2]
        self.ckpt.log[-1,3] = r[4]
        self.ckpt.log[-1,4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log('best {}'.format(best))
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f}, rank1: {:.4f}, rank3: {:.4f}, rank5: {:.4f}, rank10: {:.4f}, (Best: {:.4f} @epoch {})'.format(
                m_ap,
                r[0], r[2], r[4], r[9],
                best[0][0],
                epoch
            )
        )

        #---------注意合并模型保存开关-----------
        #self.ckpt.save(self, epoch, is_best=True)

        if not self.args.test_only and not self.args.test_all:
            m_ap_best = best[0][0].numpy()
            m_ap_best = np.around(m_ap_best, 4)

            m_ap_now = round(m_ap, 4)

            self.write_log('----{:.4f},  {}'.format(m_ap_best, m_ap_now))
            if abs(m_ap_best - m_ap_now) <= 0.0001:
                self.write_log('True')
                is_best = True
            elif m_ap_now - m_ap_best < -0.1:
                self.write_log('batch not good')
                is_best = False
            else:
                is_best = False

            self.ckpt.save(self, epoch, is_best=is_best)

            
    def extract_feature_mp(self, loader):
        features = []
        batch = 0
        for (inputs, labels) in loader:
            batch+=1
            if dist.get_rank()==0:
                print('\r' + 'batch: [' + str(batch) + '/' + str(len(loader)) + ']', end='')

            input_img = inputs.to(self.device)
            labels = labels.to(self.device)
            inp=[input, 'reid', labels]
            with torch.no_grad():
                with autocast(enabled=fp16):
                    outputs = self.model(inp)
            features.append(outputs['predict'])
        
        features = torch.cat(features, 0)
        return features.float()      
                      

    def pca(self, fea, dim=256):
        pca = decomposition.PCA(n_components=dim)
        pca.fit(fea)
        mean = pca.mean_
        A = pca.components_.T
        return mean, A


    def terminate(self):
        if self.args.test_only:
            self.test_mp()
            return True
        else:
            epoch = len(self.loss.log)
            return epoch >= self.args.epochs

