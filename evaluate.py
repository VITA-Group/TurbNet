import os
import sys
import time
import torch
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import TrainData, TestData, EvaluateData
from model.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim
from torchvision.models import vgg16
import torchvision.utils as utils
import math
import yaml
with open("config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6'
use_cuda = torch.cuda.is_available()



def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def save_image(turb_images, image_names, loc):
    turb_images = torch.split(turb_images, 1, dim=0)
    batch_num = len(turb_images)

    for ind in range(batch_num):
        # scaled_image = turb_images[ind].resize((400, 400), Image.ANTIALIAS)
        print('{}/{}'.format(loc,  '_'.join(image_names[ind].split("/")[-2:])))
        utils.save_image(turb_images[ind], '{}/{}'.format(loc, "output.png"))

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print("Directory already exist!")
        sys.exit(0)
        
def validation(net, test_data_loader, save_dir, save_tag=True):
    
    print("Testing ...")

    for batch_id, val_data in enumerate(test_data_loader):
        with torch.no_grad():
            turb, gt, image_names = val_data
            turb = turb.cuda()
            gt = gt.cuda()
            
            _, J, T, I = net(turb)
 

        # --- Save image --- #
        if save_tag:
            save_image(J, image_names, save_dir) 

    return -1, -1

if __name__ == "__main__":
   
    save_dir = cfg["output"]["output_path"]
    val_data_dir = cfg["eval_image"]["image_path"]
    model_path = cfg["models"]["model_path"]
    
    crop_size = [400, 400]
    test_batch_size = 1
    
    net = get_model()
    net = torch.nn.DataParallel(net)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    print("Model loaded Successfully")
    # print(net)
    
    test_data_loader = DataLoader(EvaluateData(val_data_dir = val_data_dir), batch_size=test_batch_size, shuffle=True, num_workers=8)
    print("DATALOADER DONE!")
    
    create_dir(save_dir)

    
    print("===> Evaluation Start ...")

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, test_data_loader, save_dir)
    print("===> Evaluation Completed ...")
    