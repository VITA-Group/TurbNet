import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
# from utils import edge_compute

class TrainData(data.Dataset):

    def __init__(self, crop_size, train_data_dir = "./Datasets/train_low"):

        super().__init__()


#         image_dir = train_data_dir + "/turb/"
#         directories = os.listdir(image_dir)
#         self.inp_filenames = []
#         self.tar_filenames = []
        
#         for x in directories[:5000]:
#             path = os.path.join(image_dir, x)
#             image_list = os.listdir(path)
#             self.inp_filenames.append(os.path.join(path, image_list[0]))
#             self.tar_filenames.append(os.path.join(path, image_list[0]).replace("turb", "gt"))

        image_dir = train_data_dir + "/turb/"
        image_list = os.listdir(image_dir)
        self.inp_filenames = []
        self.tar_filenames = []
        
        for x in image_list:
            if "png" in x:
                self.inp_filenames.append(os.path.join(image_dir, x))
                self.tar_filenames.append(os.path.join(image_dir, x).replace("turb", "gt"))
        
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir


    def __getitem__(self, index):

        crop_width, crop_height =  self.crop_size
        
        # haze_name = self.haze_names[index]
        # gt_name = haze_name.split('_')[0] + '.jpg'
        
        turb_name = self.inp_filenames[index]
        gt_name = self.tar_filenames[index]
        
        turb_img = Image.open(turb_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')

        width, height = turb_img.size

        if width < crop_width or height < crop_height:
            if width < height:
                turb_img = turb_img.resize((400, (int)(height * 400/ width)), Image.ANTIALIAS)
                gt_img = gt_img.resize((400, (int)(height * 400/ width)), Image.ANTIALIAS)
            elif width >= height:
                turb_img = turb_img.resize(((int)(width * 400/ height), 400), Image.ANTIALIAS)
                gt_img = gt_img.resize(((int)(width * 400 / height), 400), Image.ANTIALIAS)
            
            width, height = turb_img.size
        # --- random crop --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        turb_crop_img = turb_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        
        transform_turb = Compose([
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_gt = Compose([
            ToTensor()
        ])

        turb = transform_turb(turb_crop_img)
        gt = transform_gt(gt_crop_img)
        
        if list(turb.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return turb, gt


    def __len__(self):
        return len(self.inp_filenames)
    
    
class TestData(data.Dataset):
    def __init__(self, val_data_dir = "./Datasets/test_low"):
        super().__init__()
        
#         image_dir = val_data_dir + "/turb/"
#         directories = os.listdir(image_dir)
#         self.inp_filenames = []
#         self.tar_filenames = []
        
#         for x in directories[:1000]:
#             path = os.path.join(image_dir, x)
#             image_list = os.listdir(path)
#             self.inp_filenames.append(os.path.join(path, image_list[0]))
#             self.tar_filenames.append(os.path.join(path, image_list[0]).replace("turb", "gt"))

        image_dir = val_data_dir + "/turb/"
        image_list = os.listdir(image_dir)
        self.inp_filenames = []
        self.tar_filenames = []
        
        for x in image_list[:1000]:
            if "png" in x:
                self.inp_filenames.append(os.path.join(image_dir, x))
                self.tar_filenames.append(os.path.join(image_dir, x).replace("turb", "gt"))


    def get_images(self, index):
        turb_name = self.inp_filenames[index]
        gt_name = self.tar_filenames[index]
        
        turb_img = Image.open(turb_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')
        
        # haze_reshaped = haze_img
        turb_reshaped = turb_img.resize((400, 400), Image.ANTIALIAS)
        gt_reshaped = gt_img.resize((400, 400), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_turb = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        
        # transform_gt = Compose([ToTensor()])
        turb = transform_turb(turb_reshaped)
        gt = transform_gt(gt_reshaped)

        return turb, gt, turb_name
    
    def __getitem__(self, index):
        turb, gt, turb_name = self.get_images(index)
        return turb, gt, turb_name

    def __len__(self):
        return len(self.inp_filenames)
    
class EvaluateData(data.Dataset):
    def __init__(self, val_data_dir = "./Datasets/test_data"):
        super().__init__()
        
        image_dir = val_data_dir
        image_list = os.listdir(image_dir)
        self.inp_filenames = []
        self.tar_filenames = []
        
        for x in image_list:
            if "png" in x:
                self.inp_filenames.append(os.path.join(image_dir, x))
                self.tar_filenames.append(os.path.join(image_dir, x))


    def get_images(self, index):
        turb_name = self.inp_filenames[index]
        gt_name = self.tar_filenames[index]
        
        turb_img = Image.open(turb_name).convert('RGB')
        gt_img = Image.open(gt_name).convert('RGB')
        
        # haze_reshaped = haze_img
        turb_reshaped = turb_img.resize((400, 400), Image.ANTIALIAS)
        gt_reshaped = gt_img.resize((400, 400), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_turb = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])
        
        # transform_gt = Compose([ToTensor()])
        turb = transform_turb(turb_reshaped)
        gt = transform_gt(gt_reshaped)

        return turb, gt, turb_name
    
    def __getitem__(self, index):
        turb, gt, turb_name = self.get_images(index)
        return turb, gt, turb_name

    def __len__(self):
        return len(self.inp_filenames)