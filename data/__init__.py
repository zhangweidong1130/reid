from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing, letterbox_image_xy
from data.sampler import RandomSampler,DistributRandomIdCamSampler,DistrubutTestSampler
from torch.utils.data import dataloader,Dataloder

from torch.utils.data.distributed import DistributedSampler
from prefetch_generator import BackgroundGenerator

class DataLoaderx(Dataloader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Data:
    def __init__(self, args):

        train_list = [
            #transforms.Resize((args.height, args.width), interpolation=3),
            letterbox_image_xy(args.height, args.width),
        
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        ]

        train_list_kd = [
            #transforms.Resize((args.height, args.width), interpolation=3),
            letterbox_image_xy(args.height, args.width),
        
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        ]

        if args.random_erasing:
            train_list.append(RandomErasing(probability=0.3, mean=[0.0, 0.0, 0.0]))
            train_list_kd.append(RandomErasing(probability=0.3, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)
        train_transform_kd = transforms.Compose(train_list_kd)

        test_transform = transforms.Compose([
            #transforms.Resize((args.height, args.width), interpolation=3),
            letterbox_image_xy(args.height, args.width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
        ])


        self.trainset = None
        if not args.test_only and not args.test_all:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, train_transform_kd, 'train')


            self.sampler = DistributedSampler(self.trainset, shuffle=True)
            #self.sampler = DistributRandomIdCamSampler(self.trainset,args.batchid,batch_image=args.batchimage)#有道云记录
            self.train_loader = DataLoaderx(self.trainset,
                            #sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                            sampler = self.sampler,
                            #shuffle=True,
                            batch_size=args.batchid * args.batchimage,
                            num_workers=args.nThread,
                            pin_memory=True,
                            drop_last=True)
        else:
            self.train_loader = [1]
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = DataLoaderx(self.testset, sampler=DistrubutTestSampler(self.testset),
                                       batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = DataLoaderx(self.queryset, sampler=DistrubutTestSampler(self.queryset)
                                        ,batch_size=args.batchtest, num_workers=args.nThread)
        