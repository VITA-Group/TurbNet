import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import math
import glob

class RestoreData(data.Dataset):
    WIDTH_IDX=1
    HEIGHT_IDX=0
    def __init__(self, data_dir = "./data", chunk_size=(400,400)):
        super().__init__()
        
        image_list = os.listdir(data_dir)
        image_list = glob.glob(
            os.path.join(data_dir, '**'),
            recursive=True
        ) 
        self.chunk_size = chunk_size
        self.tile_configurations = []
        self.source_image_cache = {}
        
        for x in image_list:

            # TODO: This is expecting png files exclusively, need to handle video
            if x.endswith(".png") or x.endswith(".jpg"):
                self.tile_configurations.extend(
                    RestoreData.get_image_tiles(x, chunk_size)
                )

    @staticmethod
    def get_image_tiles(path, tile_size):
        img = Image.open(path)
        (width, height) = img.size
        x_tiles = int(math.ceil(width / tile_size[RestoreData.WIDTH_IDX]))
        y_tiles = int(math.ceil(height / tile_size[RestoreData.HEIGHT_IDX]))
        tiles = []
        for x in range(1, x_tiles + 1):
          for y in range(1, y_tiles + 1):
            tile_x = x * tile_size[RestoreData.WIDTH_IDX]
            tile_y = y * tile_size[RestoreData.HEIGHT_IDX]

            if tile_x >= width:
                tile_x = width

            if tile_y >= height:
                tile_y = height

            tile_spec = {
                'src_path': path,
                'image_size': [height, width],
                'tile_position': [tile_y,tile_x],
                'tile_offset': [
                    abs(min(0, height - tile_size[RestoreData.HEIGHT_IDX])),
                    abs(min(0, width - tile_size[RestoreData.WIDTH_IDX]))
                ],
                'tile_count': x_tiles * y_tiles
            }
            tiles.append(tile_spec)
        return tiles

    def get_images(self, index):
        tile_conf =  self.tile_configurations[index]
        position_y, position_x = tile_conf['tile_position']
        offset_y, offset_x = tile_conf['tile_offset']
        chunk_y, chunk_x = self.chunk_size

        if tile_conf['src_path'] not in self.source_image_cache:  # here
            self.source_image_cache[tile_conf['src_path']] = np.asarray(
                Image.open(
                    tile_conf['src_path']
                ).convert('RGB')
            )
        src_img = self.source_image_cache[tile_conf['src_path']]
        # Pillow uses sizes in x,y order, so this is only place we should
        # be using it that way
        turb_img = np.asarray(Image.new('RGB', (chunk_x, chunk_y)))
        

        turb_img[offset_y:chunk_y,
                 offset_x:chunk_x
        ] = src_img[
                (position_y-chunk_y+offset_y):position_y,
                (position_x-chunk_x+offset_x):position_x
        ]


        # --- Transform to tensor --- #
        transform_turb = Compose([ToTensor()])
        
        turb = transform_turb(turb_img)


        return turb, tile_conf
    
    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.tile_configurations)
