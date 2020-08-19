# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:40:03 2020

@author: giles
"""
import random
import h5py
import numpy as np
from PIL import Image
import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset

def dataNamePair(datadir):
    im_list = os.listdir(datadir)
    name_list = [] 
    index_list = [] 
    current_list=[] 
    current = 0
    for i in range(len(im_list)):
        name = im_list[i]
        split = name.split("_")
        if split[0] not in name_list:
            index_list.append(current_list)
            current_list=[]
            current = split[0]
            name_list.append(split[0])
        current_list.append(i)
        if i == len(im_list)-1:
            index_list.append(current_list)
    index_list.pop(0)
    
    name_pair = []
    for cur_list in index_list:
        length = len(cur_list)
        for j in range(length-1):   #index1 = cur_list[j]
            for k in range(length-1-j):    #index2 = cur_list[k]  
                name1 = im_list[cur_list[j]]
                name2 = im_list[cur_list[k+j+1]]
                name_pair.append([name1,name2])
    
    return name_pair

class ShapenetDataset(Dataset):
    def __init__(self, file_path, is_train=True):
        
        self.name_pair = dataNamePair(file_path)
        self.length = len(self.name_pair)

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        

        if self.is_train:
            source_imgs = []
            source_azimuths = []
            source_elevations = []
            azimuth = self.h5_file['Azimuths'][idx].item()
            elevation = self.h5_file['Elevations'][idx].item()
            base_index = int(idx - (azimuth * 3) - (elevation / 10))

            source_imgs.append(self.transforms(self.h5_file['Images'][idx].astype(np.uint8)))
            source_azimuths.append(azimuth * self.azimuth_increment)
            source_elevations.append(elevation)
            model_name = self.h5_file['ModelNames'][idx][0].decode()

            # Some basic math to retrieve required number of source views
            for jj in range(0, self.num_views - 1):
                azimuth = random.randint(0, self.n_azimuth_angles - 1)
                elevation = random.randint(0, self.n_elevations - 1)
                idx = base_index + azimuth * 3 + elevation
                source_imgs.append(self.transforms(self.h5_file['Images'][idx].astype(np.uint8)))
                source_azimuths.append(self.h5_file['Azimuths'][idx].item() * self.azimuth_increment)
                source_elevations.append(self.h5_file['Elevations'][idx].item())

            # Retrive a random target image
            # TODO: Replace this with a completely random transformation once our approach works
            azimuth = random.randint(0, self.n_azimuth_angles - 1)
            elevation = random.randint(0, self.n_elevations - 1)
            target_idx = base_index + azimuth * 3 + elevation
            target_azimuth = self.h5_file['Azimuths'][target_idx].item() * self.azimuth_increment
            target_elevation = self.h5_file['Elevations'][target_idx].item()
            target_img = self.h5_file['Images'][target_idx].astype(np.uint8)

            data = {'source_imgs': source_imgs,
                    'target_img': self.transforms(target_img),
                    'source_azimuths': source_azimuths,
                    'target_azimuth': target_azimuth,
                    'source_elevations': source_elevations,
                    'target_elevation': target_elevation,
                    'model_name': model_name}
        else:
            images = self.h5_file['Images'][idx]
            source_azimuth, target_azimuth = self.h5_file['Azimuths'][idx]
            source_elevation, target_elevation = self.h5_file['Elevations'][idx]
            model_name = self.h5_file['ModelNames'][idx][0].decode()

            data = {'source_imgs': self.transforms(images[0].astype(np.uint8)),
                    'target_img': self.transforms(images[1].astype(np.uint8)),
                    'source_azimuths': source_azimuth.item(),
                    'target_azimuth': target_azimuth.item(),
                    'source_elevations': source_elevation.item(),
                    'target_elevation': target_elevation.item(),
                    'model_name': model_name}
        return data

    def __len__(self):
        return self.length
