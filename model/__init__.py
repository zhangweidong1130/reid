import os
from importlib import import_module

import torch
import torch.nn as nn

import torch.distributed as dist
import numpy as np
from utils.utility import printd

class Model(nn.Module):
    def __init__(self, args, ckpt):
        super(Model, self).__init__()
        print('[INFO] Making model...')

        self.args = args
        torch.cuda.set_device(args.local_rank)

        self.device = torch.device(f'cuda:{args.local_rank}')
        self.nGPU = args.nGPU
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)


        self.load(
            ckpt.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def get_model(self):

        return self.model.module

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        print("loading model")
        torch.cuda.set_device(self.args.local_rank)
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        kwargs = {'map_location': torch.device('cpu')}
        if resume == -1:
            printd('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        elif resume == -16:#不加全连接层
            print('*************240930**************')
            print(kwargs)
            ckpt = torch.load(pre_train, **kwargs)
            new_ckpt = {}
            for key in ckpt.keys():
                #if key.startswith('head_fc'):
                if 'head_fc' in key:
                    continue
                else:
                    new_ckpt[key] = ckpt[key]
            if pre_train != '':
                print('Loading model from {}'.format(pre_train))
                self.model.load_state_dict(
                    new_ckpt,
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )