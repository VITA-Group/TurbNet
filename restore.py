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
from utils.data_loader_restore import RestoreData
from model.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim
from torchvision.models import vgg16
import torchvision.utils as utils
import math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
use_cuda = torch.cuda.is_available()
save_cache = {}

def write_image(data, save_path): 
    Image.fromarray((data*255).astype(np.uint8)).save(save_path)

def save_images(tiles, tile_confs):
    tiles = torch.split(tiles, 1, dim=0)
    for idx, data in enumerate(zip(tiles, tile_confs)):
        tile, tile_conf = data
        
        if tile_conf['src_path'] not in save_cache:
            print(f"Processing {tile_conf['src_path']}")
            save_cache[tile_conf['src_path']] = {
                'tile_number': 0,
                'tile_count': tile_conf['tile_count'],
                'data': np.zeros((
                    tile_conf['image_size'][RestoreData.HEIGHT_IDX],
                    tile_conf['image_size'][RestoreData.WIDTH_IDX],
                    3
                ))
            }
        cache = save_cache[tile_conf['src_path']]
        dat = cache['data']

        position_y, position_x = tile_conf['tile_position']
        offset_y, offset_x = tile_conf['tile_offset']
        chunk_y, chunk_x = crop_size

        local_tile = tile.cpu().numpy()[0]
        outblock = np.moveaxis(local_tile, 0,2)
        outblock = outblock[ offset_y:chunk_y, offset_x:chunk_x, : ]
 
        dat[position_y-chunk_y+offset_y:position_y,
            position_x-chunk_x+offset_x:position_x,
            :
        ] = outblock

        cache['tile_number'] += 1
        print(f"\rtile {cache['tile_number']} of {cache['tile_count']}", end="", flush=True)
        if cache['tile_number'] == cache['tile_count']:
            if args.outdir is None:
                base, extension = os.path.splitext(tile_conf['src_path'])
                save_path = os.path.join(loc, f"{base}_restored{extension}")
            else:
                rel_path = os.path.relpath(tile_conf['src_path'], args.datadir)
                save_path = os.path.join(args.outdir, rel_path)
                output_dir = os.path.dirname(save_path)
                os.makedirs(output_dir, exist_ok=True)
            print(f"\rWriting {save_path}                ")
            write_image(dat, save_path)
            cache = None
            save_cache.pop(tile_conf['src_path'])
        


def restore(net, restore_data, save_tag=True):
    print("Processing ...")

    for batch_id, val_data in enumerate(restore_data):
        tiles, confs = val_data
        with torch.no_grad():
            tiles = tiles.cuda()
            
            _, J, T, I = net(tiles)
            
        # --- Save image --- #
        if save_tag:
            save_images(tiles, confs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restore image')
    parser.add_argument('--weights', type=str, default="turb_current37.pth",
        help="weights to use. default=turb_current37.pth"
    )
    parser.add_argument('datadir', type=str, 
        help="Input directory to process pngs in."
    )
    parser.add_argument('--cropsize', type=int, default=256,
        help="Image crop size."
    )
    parser.add_argument('--outdir', type=str, default=None,
        help="Output director to write restored files to. Defaults to datadir"
    )

    args = parser.parse_args()

    crop_size = (args.cropsize, args.cropsize)
    train_batch_size = 12
    test_batch_size = 1
    num_epochs = 50
    gps=3
    blocks=19
    lr=1e-4
    all_T = 100000
    old_val_psnr = 0
    
    #net = get_model(gps, blocks)
    net = get_model()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join("./weights",args.weights)))
    print("Model loaded Successfully")
    
    net.cuda()
   
    def collate_fn(data):
        tiles = torch.stack([t[0] for t in data])
        confs = [t[1] for t in data]
        return tiles, confs

    restore_data = DataLoader(
        RestoreData(data_dir=os.path.abspath(args.datadir), chunk_size=crop_size),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1
    )
    print("DATALOADER DONE!")

    print("===> Evaluation Start ...")

    # --- Use the evaluation model in testing --- #
    net.eval()

    restore(net, restore_data)
    
