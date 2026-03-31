import torch.utils.data
from data.base_data_loader import BaseDataLoader
import pdb

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'single':  # yes
        from data.single_dataset import SingleDataset   
        dataset = SingleDataset()
    elif (opt.dataset_mode == 'aligned') or (opt.dataset_mode == 'aligned'):
        raise ValueError('In NLOS-OT, we only support dataset in single mode.')
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))    
    dataset.initialize(opt)     
    return dataset 


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt) 
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            pin_memory=False,  # 修改为False，减少内存使用
            num_workers=int(opt.nThreads),
            persistent_workers=False,  # 修改为False，避免worker持久化
            prefetch_factor=2,  # 添加预取因子，平衡内存和性能
            drop_last=True  # 丢弃最后一个不完整的batch，避免内存泄漏
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def close(self):
        """手动关闭数据加载器"""
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'close_file_handles'):
            self.dataset.close_file_handles()