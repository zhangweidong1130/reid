import random
import collections
from torch.utils.data import sampler
import torch.distributed as dist
import math

class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
    

class DistrubutTestSampler(sampler.Sampler):

    def __init__(self, data_source, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.data_source) * 1.0 / self.num_replicas) ) 
        

    def __iter__(self):

        indices = list(range(len(self.data_source))) 

        indices = indices[self.rank * self.num_samples:(self.rank + 1)* self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples