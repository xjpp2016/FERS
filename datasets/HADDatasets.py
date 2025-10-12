import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.gen_mask import Mask
from skimage import transform as sk_transform
import random


class HADDataset(Dataset):
    def __init__(self,
                 dataset_path='./',
                 sensor = 'aviris_ng',
                 resize=64,
                 start_channel=0,
                 channel = 50,
                 train_ratio = 1
                 ):
        self.dataset_path = dataset_path
        self.resize = resize
        self.start_channel = start_channel
        self.channel =channel
        self.sensor =sensor
        self.train_ratio = train_ratio
        # load dataset
        self.train_img,  self.paste_img = self.load_dataset_folder()
        # set transforms
        self.transform= transforms.Compose([
            transforms.ToTensor()])
    def __getitem__(self, idx):
        # load image
        img_path= self.train_img[idx]
        x = np.load(img_path)

        x=x[:,:,self.start_channel:(self.channel+self.start_channel)]
        x = (x-np.min(x)) / (np.max(x)-np.min(x))*2-1
        x = x*0.1

        x = self.transform(x)
        x = x.type(torch.FloatTensor)

        return x

    def __len__(self):
        return len(self.train_img)

    def load_dataset_folder(self):
        if self.sensor == 'aviris_ng' or self.sensor =='all':
            train_img_dir = os.path.join(self.dataset_path, 'train','aviris_ng')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris')
        elif self.sensor == 'aviris':
            train_img_dir = os.path.join(self.dataset_path, 'train','aviris')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris_ng')
        else:
            train_img_dir = os.path.join(self.dataset_path, 'test','aviris_ng')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris')
        train_list = sorted(
            [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.npy')])
        train_list = train_list[:int(len(train_list)* self.train_ratio)]
        paste_list = sorted(
            [os.path.join(paste_img_dir, f) for f in os.listdir(paste_img_dir) if f.endswith('.npy')])
        if self.sensor == 'all':
            train_list = train_list + paste_list

        return train_list, paste_list


class HADTestDataset(Dataset):
    def __init__(self,
                 dataset_path='./',
                 resize=64,
                 start_channel=0,
                 channel = 100
                 ):
        self.dataset_path = dataset_path
        self.resize = resize
        self.start_channel = start_channel
        self.channel =channel
        self.sensor = 'aviris_ng'

        # load dataset
        self.test_img, self.gt_img= self.load_dataset_folder()

        # set transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, idx):
        x, gt= self.test_img[idx], self.gt_img[idx]
        # load test image
        x = np.load(x)
        x = x[:, :, self.start_channel:(self.channel + self.start_channel)]
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        x = x * 0.1
        x = self.transform(x)
        x = x.type(torch.FloatTensor)

        # load gt
        gt = Image.open(gt)
        gt =np.array(gt)
        gt = gt[:, :, 1]
        gt = Image.fromarray(gt)
        gt = self.transform(gt)
        return x,gt

    def __len__(self):
        return len(self.test_img)

    def load_dataset_folder(self):
        test_img_dir = os.path.join(self.dataset_path, 'test', self.sensor)
        gt_dir = os.path.join(self.dataset_path, 'ground_truth',self.sensor)
        test_img = sorted(
            [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith('.npy')])
        img_name_list = [os.path.splitext(os.path.basename(f))[0] for f in test_img]
        gt_img = [os.path.join(gt_dir, img_name + '.png') for img_name in img_name_list]
        assert len(test_img) == len(gt_img), 'number of test img and gt should be same'
        return test_img, gt_img



class HADDataset_Aug(Dataset):
    def __init__(self,
                 dataset_path='./',
                 sensor = 'aviris_ng',
                 mask_class = 'zero',
                 mask_type = 'none',
                 resize=64,
                 start_channel=0,
                 channel = 50,
                 train_ratio = 1
                 ):
        self.dataset_path = dataset_path
        self.mask_class = mask_class
        self.mask_type = mask_type
        self.resize = resize
        self.start_channel = start_channel
        self.channel =channel
        self.sensor =sensor
        self.train_ratio = train_ratio
        self.mask_generator =Mask(resize=self.resize,)
        # load dataset
        self.train_img,  self.paste_img = self.load_dataset_folder()
        # set transforms
        self.transform= transforms.Compose([
            transforms.ToTensor()])
        
        self.has_no = True    

    def __getitem__(self, idx):
        # load image
        img_path= self.train_img[idx]
        x = np.load(img_path)
        x=x[:,:,self.start_channel:(self.channel+self.start_channel)]
        x = (x-np.min(x)) / (np.max(x)-np.min(x))*2-1
        x = x*0.1

        # rotate
        x = sk_transform.rotate(x,random.choice([0,90,180,270]))
        # flip
        if random.random()>0.5:
            x=x[:,::-1, ...].copy()
        if random.random()>0.5:
            x=x[::-1, ...].copy()
        # add mask
        mask = self.mask_generator(1)[0]
        mask = np.expand_dims(mask, axis=2)
        x = self.transform(x)
        x = x.type(torch.FloatTensor)
        mask = self.transform(mask)
        mask = mask.type(torch.FloatTensor)
        # mask_out = (0.2 * np.random.rand(x.shape[0], x.shape[1], x.shape[2]) - 0.5)
        # mask_out = 0*np.ones([x.shape[0], x.shape[1], x.shape[2]])

        if self.mask_type=='none':
            pass
        elif self.mask_type=='random':
            if self.has_no:
                self.mask_class = random.choice(['no','zero','random','sin'])
            else:
                self.mask_class = random.choice(['zero','random','sin'])
            # print('mask class is {}'.format(self.mask_class))

        if self.mask_class == 'no':
            x_m = x
            mask = 1 - mask*0
        elif self.mask_class == 'zero':
            x_m = mask * x
        elif self.mask_class == 'random':
            paste = np.random.rand(x.shape[0], x.shape[1], x.shape[2])
            paste = (paste * 2 - 1) * 0.1
            paste = self.transform(paste)
            paste = paste.permute(1, 0, 2)
            x_m = mask * x + (1 - mask) * paste
        elif self.mask_class == 'sin':
            sin_head = np.pi*np.random.rand(1)
            sin = np.sin(np.linspace(sin_head,sin_head+ 2 * np.pi, self.channel))
            paste = sin.reshape(self.channel,1,1).repeat(self.resize ,axis=1).repeat(self.resize ,axis=2)
            paste = paste*0.1
            paste = self.transform(paste)
            paste = paste.permute(1, 0, 2)            
            x_m = mask * x + (1 - mask) * paste
        else:
            raise Exception("this mode is not defined")
        # x_m_o = mask * x + (1 - mask) * mask_out
        x_m = x_m.type(torch.FloatTensor)
        # x_m_o = x_m_o.type(torch.FloatTensor)
        return x, x_m, 1-mask, self.mask_class

    def __len__(self):
        return len(self.train_img)

    def load_dataset_folder(self):
        if self.sensor == 'aviris_ng' or self.sensor =='all':
            train_img_dir = os.path.join(self.dataset_path, 'train','aviris_ng')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris')
        elif self.sensor == 'aviris':
            train_img_dir = os.path.join(self.dataset_path, 'train','aviris')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris_ng')
        else:
            train_img_dir = os.path.join(self.dataset_path, 'test','aviris_ng')
            paste_img_dir = os.path.join(self.dataset_path, 'train','aviris')
        train_list = sorted(
            [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.npy')])
        # train_list = sorted(
        #     [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.npy') and
        #                                       'ang20170821t183707' in f])
        train_list = train_list[:int(len(train_list)* self.train_ratio)]
        paste_list = sorted(
            [os.path.join(paste_img_dir, f) for f in os.listdir(paste_img_dir) if f.endswith('.npy')])
        if self.sensor == 'all':
            train_list = train_list + paste_list

        return train_list, paste_list

