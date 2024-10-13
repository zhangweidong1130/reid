import os
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
from utils.nadam import Nadam
from utils.n_adam import NAdam
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

import torch
import torch.nn as nn
from PIL import Image

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '':
            if args.save == '': args.save = now
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = ''
            else:
                self.log = torch.load(self.dir + '/map_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)*args.test_every))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)
        
        if dist.get_rank() == 0:

            _make_dir(self.dir)
            _make_dir(self.dir + '/model')
            #_make_dir(self.dir + '/results')

            open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
            self.log_file = open(self.dir + '/log.txt', open_type)
            with open(self.dir + '/config.txt', open_type) as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

    def save(self, trainer, epoch, train_save=False, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        torch.save(self.log, os.path.join(self.dir, 'map_log.pt'))       
        if not train_save:
            self.plot_map_rank(epoch)

        
        # torch.save(
        #     trainer.optimizer.state_dict(),
        #     os.path.join(self.dir, 'optimizer.pt')
        # )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False, end='\n'):
        if dist.get_rank() == 0:       
            print(log, end=end)
            if end != '':
                self.log_file.write(log + end)
            if refresh:
                self.log_file.close()
                self.log_file = open(self.dir + '/log.txt', 'a')

    def printd(*string):
        if dist.get_rank() == 0:
            print(*string)
    
    def done(self):
        self.log_file.close()

    def plot_map_rank(self, epoch):
        axis = np.linspace(1, epoch, self.log.size(0))
        label = 'Reid on {}'.format(self.args.data_test)
        labels = ['mAP','rank1','rank3','rank5','rank10']
        fig = plt.figure()
        plt.title(label)
        for i in range(len(labels)):
            plt.plot(axis, self.log[:, i].numpy(), label=labels[i])

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('mAP/rank')
        plt.grid(True)
        plt.savefig('{}/test_{}.jpg'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum )/ (self.count + 1e-20)



def make_optimizer_vit(args, model):
    print('init optimizer')

    for param in model.parameters():
        param.requires_grad = True
    
    # for param in model.model.head_fc.parameters():
    #     param.requires_grad = True
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     if 'lora' in name:
    #         param.requires_grad = True
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    regularized = []
    not_regularized = []
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    trainable = [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    return optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {
            'momentum': args.momentum,
            'dampening': args.dampening,
            'nesterov': args.nesterov
            }
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'amsgrad': args.amsgrad
        }
    elif args.optimizer == 'NADAM':
        optimizer_function = NAdam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {
            'eps': args.epsilon,
            'momentum': args.momentum
        }
    else:
        raise Exception()

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, optimizer, len_dataloader=None):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif arg.decay_type == 'Plateau':
        scheduler = lrs.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=20,
            verbose=True,
        )
    elif args.decay_type == 'Cos':
        scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=5*len_dataloader,
            eta_min=1e-4,
            last_epoch=-1
        )

    return scheduler

def printd(*string):
    if dist.get_rank() == 0:
        print(*string)

def adjust_lr_fc(optimizer, ep, total_ep, start_decay_at_ep):
    if ep <=10:
        lr = 0.1
    elif ep <=20:
        lr = 0.03
    elif ep <=30:
        lr = 0.01
    elif ep <=40:
        lr = 0.003
    elif ep <=50:
        lr = 0.001
    elif ep <=60:
        lr = 0.0003
    elif ep <=100:
        lr = 0.0001
    elif ep <=200:
        lr = 0.00001
    else:
        return
    
    for p in optimizer.param_groups:
        p['lr'] = lr


def adjust_lr_vit(optimizer, ep, total_ep, start_decay_at_ep):
    # if ep <=10:
    #     lr = np.linspace(0.00001, 0.001, 10)[ep-1]
    # elif ep <=20:
    #     lr = 0.001
    # elif ep <=30:
    #     lr = 0.0005
    # elif ep <=40:
    #     lr = 0.00025
    # elif ep <=50:
    #     lr = 0.0001
    # elif ep <=60:
    #     lr = 0.00005
    # elif ep <=70:
    #     lr = 0.00002
    # elif ep <=200:
    #     lr = 0.00001
    # else:
    #     return
    
    # for p in optimizer.param_groups:
    #     p['lr'] = lr

    if ep <=10:
        lr = 0.0001
    elif ep <=20:
        lr = 0.00005
    elif ep <=20:
        lr = 0.00002
    elif ep <=200:
        lr = 0.00001
    else:
        return
    
    for p in optimizer.param_groups:
        p['lr'] = lr

def recursive_to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, dict):
        return {k: recursive_to_device(v, device) for k, v in input.items()}
    elif isinstance(input, list):
        return [recursive_to_device(x, device) for x in input]
    elif isinstance(input, tuple):
        return tuple(recursive_to_device(x, device) for x in input)
    

def Save_Testerro(data_test, erro_dict):#有道云记录
    pass