import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageOps
import re
import pdb
import cv2
import numpy as np

MAX_VALUE = 100

def update(img, brightness,saturation):
    """
    用于修改图片的亮度和饱和度
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    :param lightness: 亮度
    :param saturation: 饱和度
    """
    Imax = np.max(img)
    Imin = np.min(img)
    MAX = 255
    MIN = 0
    ime = (img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
    ime = img.astype('uint8')
    max_percentile_pixel, min_percentile_pixel = compute(ime, 1, 99)
    # 去掉分位值区间之外的值
    ime[ime >= max_percentile_pixel] = max_percentile_pixel
    ime[ime <= min_percentile_pixel] = min_percentile_pixel
    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    im = np.zeros(ime.shape, ime.dtype)
    cv2.normalize(ime, im, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    # 计算亮度
    hsv_image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    #print(lightness)

    if 150 < lightness:
        return im
        # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
        # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    elif 80 <= lightness <= 150:
        im = im.astype(np.float32) / 255.0
        # 调整亮度
        hlsImg = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
        # 1.调整亮度（线性变换)
        hlsImg[:, :, 1] = (1.0 + brightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
        hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
        # 饱和度
        hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
        hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        out = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
        out = out.astype(np.uint8)
        return out
    elif lightness < 80:
        im = im.astype(np.float32) / 255.0
        # 调整亮度
        hlsImg = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
        # 1.调整亮度（线性变换)
        hlsImg[:, :, 1] = (1.0 + brightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
        hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
        # 饱和度
        hlsImg[:, :, 2] = (0.5 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
        hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
        lsImg = lsImg.astype(np.uint8)
        return lsImg

def compute(ime, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(ime, max_percentile)
    min_percentile_pixel = np.percentile(ime, min_percentile)
    return max_percentile_pixel, min_percentile_pixel


def resize_img_keep_ratio(img,target_size):
    img = np.array(img)
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new.astype('uint8')


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        if opt.phase=='train':
            self.opt = opt
            self.root = opt.datarootTarget
            self.dir_B = os.path.join(opt.datarootTarget)
            self.dir_A = os.path.join(opt.datarootData)
            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)

            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)


            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list) # data/custom_dataset_data_loader.py(21)CreateDataset
        elif opt.phase == 'val':
            self.opt = opt
            self.root = opt.datarootTarget
            self.dir_B = os.path.join(opt.datarootValTarget)
            self.dir_A = os.path.join(opt.datarootValData)
            self.A_paths = make_dataset(self.dir_A)
            self.B_paths = make_dataset(self.dir_B)
            self.A_paths = sorted(self.A_paths)
            self.B_paths = sorted(self.B_paths)


            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list) # data/custom_dataset_data_loader.py(21)CreateDataset
        elif opt.phase == 'test':
            self.opt = opt
            self.root = opt.datarootData
            self.dir_A = os.path.join(opt.datarootData)
            self.A_paths = make_dataset(self.dir_A)
            self.A_paths = sorted(self.A_paths)

            self.dir_B = os.path.join(opt.datarootTarget)
            self.B_paths = make_dataset(self.dir_B)
            self.B_paths = sorted(self.B_paths)
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

            self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # load input images
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((96, 96), Image.BICUBIC)
        A_img = self.transform(A_img)

        # load gt
        """
        B_path = A_path.replace(r'E:\dragon\data\imagenet-s\ImageNetS50\pro6',r'E:\dragon\data\imagenet-s\ImageNetS50')
        B_path = B_path.replace(r'_0.JPEG',r'.JPEG')
        """
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        
        brightness = 60# 亮度
        saturation = 50# 饱和度
        #B_img = update(np.array(B_img), brightness, saturation)
        B_img = resize_img_keep_ratio(B_img, [96,96])
        B_img = Image.fromarray(B_img)
        #B_img = B_img.resize((256, 256), Image.BICUBIC)
        B_img = self.transform(B_img)
        #print("*******************", A_path, B_path)

        return {'A': A_img, 'A_paths': A_path,'B': B_img, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
