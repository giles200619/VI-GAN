# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:04:04 2020

@author: giles
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from network_VIGAN import PerceptualVGG19
from torch.autograd import Variable
import torch.autograd as autograd



def get_object_T_camera(x: float, y: float, z: float) -> np.ndarray:
    z_vector = np.array([-x, -y, -z])
    e_z = z_vector / np.linalg.norm(z_vector)
    x_vector = np.cross(e_z, np.array([0,0,1]))
    e_x = x_vector / np.linalg.norm(x_vector)
    e_y = np.cross(e_z, e_x)

    camera_position = np.array([x,y,z])

    object_T_camera = np.c_[e_x, e_y, e_z, camera_position]
    return object_T_camera

def spherical_to_cartesian(azimuth: float, elevation: float, distance: float = 1.0):
    #if azimuth > 2 * np.pi or elevation > 2 * np.pi:
        #warnings.warn('Expects radians, received {} for azimuth and {} for elevation'.format(azimuth, elevation))
    z = distance * np.sin(elevation)

    d_cos = distance * np.cos(elevation)
    x = d_cos * np.cos(azimuth)
    y = d_cos * np.sin(azimuth)

    return x, y, z

def pose_from_filename(filename: str) -> np.ndarray:
    azimuth_degree, elevation_degree = tuple(float(v) for v in filename.split('.')[0].split('_')[-2:])
    azimuth_degree *= -10

    azimuth_rad, elevation_rad = np.deg2rad(azimuth_degree), np.deg2rad(elevation_degree)
    x, y, z = spherical_to_cartesian(azimuth_rad, elevation_rad)

    object_T_camera = get_object_T_camera(x, y, z)
    return object_T_camera
    
def RotationMatrix6D(img1,img2):
    R01 = pose_from_filename(img1)
    R02 = pose_from_filename(img2)
    R12 = np.transpose(R01[:,0:3]) @ R02[:,0:3]
    
    T12 = np.subtract(R02[:,3],R01[:,3])
    
    return np.hstack((np.reshape(R12[:,0:2],(1,6)),np.reshape(T12,(1,3))))

def dataNamePair(datadir):
    im_list = os.listdir(datadir)
    name_list = [] 
    index_list = [] 
    current_list=[] 
    for i in range(len(im_list)):
        name = im_list[i]
        split = name.split("_")
        if split[0] not in name_list:
            index_list.append(current_list)
            current_list=[]
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

def groundTruthTensor(filename_list):
    gt = np.zeros((len(filename_list),9))
    for i in range(len(filename_list)):
        gt[i,:] = RotationMatrix6D(filename_list[i][0],filename_list[i][1])
    
    return torch.from_numpy(gt)

def imgPreprocess(filename_list,args): #normalize
    trans = transforms.Compose([
	transforms.Resize((128,128)), 
	transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
    tensor = torch.tensor(())
    image1 = tensor.new_zeros((args.batch_size, 3,128,128))
    image2 = tensor.new_zeros((args.batch_size, 3,128,128))
    poseB = tensor.new_zeros((args.batch_size, 12))
    poseA = tensor.new_zeros((args.batch_size, 12))
    for l in range(len(filename_list)):
        if args.train == 0:
            path = args.test_dir
        else: 
            path = args.train_dir
        img1 = Image.open(os.path.join(path, filename_list[l][0]))
        img2 = Image.open(os.path.join(path, filename_list[l][1]))
        img1 = trans(img1)
        image1[l] = img1
        img2 = trans(img2)
        image2[l] = img2
        
        R1 = pose_from_filename(filename_list[l][0])
        poseA[l] = torch.from_numpy(np.reshape(R1,(12,1)))[:,0]
        R2 = pose_from_filename(filename_list[l][1])
        poseB[l] = torch.from_numpy(np.reshape(R2,(12,1)))[:,0]
    
    return image1,image2,poseA, poseB

def catImgPose(img,pose):
    pose = pose[:,:,None,None]
    pose = pose.repeat(1,1,128,128)
    input1 = torch.cat((img,pose),dim=1)
    
    return input1

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(0.0))
        self.register_buffer('fake_label', torch.tensor(1.0))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def calc_gradient_penalty(netD, real_data, fake_data,args,device):
    # ||gradient D(x)||-1  
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size, int(real_data.nelement()/args.batch_size)).contiguous().view(args.batch_size, 3, 128, 128)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def tensor2im(input_image, imtype=np.uint8):
    sflag = False
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 2:
        sflag = True
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.round((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0)
    if sflag:
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.perceptual_loss_module = PerceptualVGG19(feature_layers=[0, 5, 10, 15], use_normalization=False)

    def forward(self, input, target):
        fake_features = self.perceptual_loss_module(input)
        real_features = self.perceptual_loss_module(target)
        vgg_tgt = ((fake_features - real_features) ** 2).mean()
        return vgg_tgt
