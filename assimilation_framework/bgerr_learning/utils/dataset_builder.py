from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from assimilation_framework.bgerr_learning.dataset.era5_lr_dataset import era5_128x256
from assimilation_framework.utils.misc import get_rank, get_world_size

class DatasetBuilder(object):
    def __init__(self, params):
        super(DatasetBuilder, self).__init__()
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})
    
    def get_sampler(self, dataset, split = 'train'):
        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        rank = get_rank()
        num_gpus = get_world_size()
        sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle, num_replicas=num_gpus, seed=0)

        return sampler

    def get_dataloader(self, split='train', batch_size=None):
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 32)
            elif split == "test":
                batch_size = self.trainer_params.get('test_batch_size', 1)
            else:
                batch_size = self.trainer_params.get('valid_batch_size', 1)

        dataset = era5_128x256(split=split, **self.dataset_params[split])
        sampler = self.get_sampler(dataset, split)
        return DataLoader(dataset, batch_size = batch_size, sampler=sampler, **self.dataloader_params)
