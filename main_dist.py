import data
import loss
import torch
import model
import time
from importlib import import_module
from trainer import Trainer

from option import args
import utils.utility as utility
import numpy as np
import random
import os

import torch.distributed as dist


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dist.init_process_group(backend='nccl', init_method='env://')
# dist.init_process_group(backend='nccl', 
# 						#init_method='env://',
# 						init_method = 'tcp://127.0.0.1:4205',
# 						world_size=4,
# 						rank=args.local_rank
# 						)

torch.cuda.set_device(args.local_rank)
print('local_rank: ',args.local_rank)


if not args.test_all:
	ckpt = utility.checkpoint(args)

	loader = data.Data(args)
	model = model.Model(args, ckpt)
	loss = loss.Loss(args, ckpt) if not args.test_only else None
	module = import_module(args.trainer)

	trainer = getattr(module, 'Trainer')(args, model, loss, loader, ckpt)

	n = 0
	while not trainer.terminate():
		n += 1
		start=time.time()

		trainer.train()

		end=time.time()
		utility.printd('epoch time ---', end-start)
		utility.printd('\n')
		if args.test_every!=0 and n%args.test_every==0:
			trainer.test_mp()

else:
	print('-----------start test-------------')
	ckpt = utility.checkpoints(args)
	model = model.Model(args, ckpt)
	loss = None
	lo1ader = data.Data(args)
	module = import_module(args.trainer)

	trainer = getattr(module, 'Trainer')(args, model, loss, loader, ckpt)

	test_dataset = ['kangzhuang_night', 'day_night_test', 'huazhi_test']
	#test_dataset = ['train_person_kd_getfearure']

	for dataset in test_dataset:
		args.dataset = dataset
		loader = data.Data(args)
		trainer.loader = loader
		trainer.test_loader = loader.test_loader
		trainer.query_loader = loader.query_loader
		trainer.testset = loader.testset
		trainer.queryset = loader.queryset

		trainer.test_mp()
		#trainer.test_mp_getfeat()#特征提取

#--------多帧测试----------
	# test_dataset_multi = ['huanzhuang_lidezhi_411_multi', 'huanzhuang_lidezhi_469_multi']
	# for dataset in test_dataset_multi:
	# 	args.dataset = dataset
	# 	loader = data.Data(args)
	# 	trainer.loader = loader
	# 	trainer.test_loader = loader.test_loader
	# 	trainer.query_loader = loader.query_loader
	# 	trainer.testset = loader.testset
	# 	trainer.queryset = loader.queryset

	# 	trainer.test_mp_multi()
