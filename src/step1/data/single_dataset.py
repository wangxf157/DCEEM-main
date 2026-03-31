import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import re
# import pdb

class SingleDataset(BaseDataset):
    def initialize(self, opt):
        # pdb.set_trace()
        if opt.phase=='train':
            self.opt = opt
            self.root = opt.datarootTarget
            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.Resize((96, 96)),  # 保持96×96分辨率
                             transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        elif opt.phase == 'val':
            self.opt = opt
            self.root = opt.datarootTarget

            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootValData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootValTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.Resize((96, 96)),  # 保持96×96分辨率
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        elif opt.phase == 'test':
            self.opt = opt
            self.root = opt.datarootTarget
            self.A_paths = []
            self.B_paths = []

            self.dir_A = re.split(',',opt.datarootData)
            for i in range(len(self.dir_A)):
                dir_A = self.dir_A[i]
                Apath = make_dataset(dir_A)
                Apath = sorted(Apath)
                self.A_paths = self.A_paths + Apath

            self.dir_B = re.split(',',opt.datarootTarget)
            for i in range(len(self.dir_B)):
                dir_B = self.dir_B[i]
                Bpath = make_dataset(dir_B)
                Bpath = sorted(Bpath)
                self.B_paths = self.B_paths + Bpath
            transform_list = [transforms.Resize((96, 96)),  # 保持96×96分辨率
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

        else:
            raise ValueError('The phase must be train, val or test.')
        
        # 添加文件句柄管理
        self._file_handles = {}

    def __getitem__(self, index):
        # load groundtruth
        B_path = self.B_paths[index]
        
        # 使用with语句确保文件正确关闭
        with Image.open(B_path) as img:
            B_img = img.convert('RGB')
        
        # 这一步是为了和train时的resize一致，但是val和test时不需要resize，如果不需要resize（图像大小为256*256），注释掉这一步
        #B_img = B_img.resize((256, 256), Image.BICUBIC)
        B_img = self.transform(B_img)

        """
        # load input
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((256, 256), Image.BICUBIC)
        A_img = self.transform(A_img)
        """
        A_path = B_path
        A_img = B_img

        return {'A': A_img, 'A_paths': A_path,'B': B_img, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
    
    def close_file_handles(self):
        """手动关闭所有文件句柄"""
        if hasattr(self, '_file_handles'):
            for handle in self._file_handles.values():
                if hasattr(handle, 'close'):
                    handle.close()
            self._file_handles.clear()