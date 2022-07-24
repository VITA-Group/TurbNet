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
from utils.data_loader import TrainData, TestData
from model.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim
from torchvision.models import vgg16
import torchvision.utils as utils
import math


os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()



def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

def save_image(turb_images, image_names, loc):
    turb_images = torch.split(turb_images, 1, dim=0)
    batch_num = len(turb_images)

    for ind in range(batch_num):
        # scaled_image = turb_images[ind].resize((400, 400), Image.ANTIALIAS)
        utils.save_image(turb_images[ind], '{}/{}'.format(loc,  '_'.join(image_names[ind].split("/")[-2:])))

def create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir + "/turb")
        os.mkdir(save_dir + "/gt")
        os.mkdir(save_dir + "/T")
        os.mkdir(save_dir + "/I")
        os.mkdir(save_dir + "/J")
    else:
        print("Directory already exist!")
        sys.exit(0)
        
def validation(net, test_data_loader, save_dir, save_tag=True):
    
    print("Testing ...")
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(test_data_loader):
        with torch.no_grad():
            turb, gt, image_names = val_data
            turb = turb.cuda()
            gt = gt.cuda()
            
            _, J, T, I = net(turb)
            
        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(J, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(ssim(J, gt))

        # --- Save image --- #
        if save_tag:
            save_image(turb, image_names, save_dir + "/turb")
            save_image(gt, image_names, save_dir + "/gt")
            save_image(J, image_names, save_dir + "/J") 
            save_image(T, image_names, save_dir + "/T")
            save_image(I, image_names, save_dir + "/I")
            

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

if __name__ == "__main__":
    crop_size = [400, 400]
    train_batch_size = 6
    test_batch_size = 1
    num_epochs = 50
    gps=3
    blocks=19
    lr=1e-4
    all_T = 100000
    old_val_psnr = 0
    alpha = 0.9
    save_dir = "./current_run"
    net = get_model()
    net = torch.nn.DataParallel(net)
    net.cuda()
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_data_loader = DataLoader(TrainData(crop_size), batch_size=train_batch_size, shuffle=True, num_workers=8)
    test_data_loader = DataLoader(TestData(), batch_size=test_batch_size, shuffle=True, num_workers=8)
    print("DATALOADER DONE!")
    
    create_dir(save_dir)

    print("===> Training Start ...")
    for epoch in range(num_epochs):
        psnr_list = []
        start_time = time.time()

        # --- Save the network parameters --- #
        torch.save(net.state_dict(), '{}/turb_current{}.pth'.format(save_dir, epoch))

        for batch_id, train_data in enumerate(train_data_loader):
            if batch_id > 5000:
                break
            step_num = batch_id + epoch * 5000 + 1
            lr=lr_schedule_cosdecay(step_num,all_T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            turb, gt = train_data
            turb = turb.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            _, J, T, I = net(turb)
            Rec_Loss1 = F.smooth_l1_loss(J, gt)
            Rec_Loss2 = F.smooth_l1_loss(I, turb)
            loss = alpha * Rec_Loss1 + (1 - alpha) * Rec_Loss2

            loss.backward()
            optimizer.step()

            # --- To calculate average PSNR --- #
            psnr_list.extend(to_psnr(J, gt))

            if not (batch_id % 100):
                print('Epoch: {}, Iteration: {}, Loss: {:.3f}, Rec_Loss1: {:.3f}, Rec_loss2: {:.3f}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)
        print("Train PSNR : {:.3f}".format(train_psnr))

        # --- Use the evaluation model in testing --- #
        net.eval()

        val_psnr, val_ssim = validation(net, test_data_loader, save_dir)
        one_epoch_time = time.time() - start_time
        print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, "train", save_dir)
    
    
