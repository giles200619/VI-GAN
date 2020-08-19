# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:12:50 2020

@author: giles
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import RotationMatrix6D, 


class ImagePairDataset(Dataset):
    def __init__(self, dir_path):
        self.name_pair = dataNamePair(dir_path)
        self.length = len(self.name_pair)
        self.path = dir_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = {}
        
        filename_list = self.name_pair[index]
        img1, img2, poseA, poseB = self.imgPreprocess(filename_list)
        groundtruth_pose = self.groundTruthTensor(filename_list)
        input1 = self.catImgPose(img1,poseA)
        
        data = {'input1': input1,
                'img1':img1,
                'img2':img2,
                'poseA':poseA,
                'poseB':poseB,
                'groundtruth_pose':groundtruth_pose
                }
        
        return data
    
    def dataNamePair(self,datadir):
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

    def imgPreprocess(self,filename_list): #normalize
    
        trans = transforms.Compose([
    	transforms.Resize((128,128)), 
    	transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    	])
        tensor = torch.tensor(())
        image1 = tensor.new_zeros((1, 3,128,128))
        image2 = tensor.new_zeros((1, 3,128,128))
        poseB = tensor.new_zeros((1, 12))
        poseA = tensor.new_zeros((1, 12))
        
        image1[0] = trans(Image.open(os.path.join(self.path, filename_list[0][0])))
        image2[0] = trans(Image.open(os.path.join(self.path, filename_list[0][1])))
        
        R1 = pose_from_filename(os.path.splitext(source_img)[0])
        poseA[0] = torch.from_numpy(np.reshape(R1,(12,1)))[:,0]
        R2 = pose_from_filename(os.path.splitext(target_img)[0])
        poseB[0] = torch.from_numpy(np.reshape(R2,(12,1)))[:,0]
        
        return image1,image2,poseA, poseB

    def groundTruthTensor(self,filename_list):
        gt = np.zeros((len(filename_list),9))
        for i in range(len(filename_list)):
            gt[i,:] = RotationMatrix6D(filename_list[i][0],filename_list[i][1])
        
        return torch.from_numpy(gt)
    
    def catImgPose(self,img,pose):
        pose = pose[:,:,None,None]
        pose = pose.repeat(1,1,128,128)
        input1 = torch.cat((img,pose),dim=1)
        
        return input1