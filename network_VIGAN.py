# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:34:01 2020

@author: giles
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models
from helpers import get_norm_layer, get_nonlinear_layer
from coord_conv import CoordConv2d, CoordConvTranspose2d

def Conv2d(in_filters, out_filters, kernel_size=3, stride=1,
           padding=0, dilation=1, groups=1, bias=True, coord_conv=True):
    if coord_conv:
        return CoordConv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, bias=bias)
    
def ConvTranspose2d(in_filters, out_filters, kernel_size=3, stride=1,
                    padding=0, dilation=1, output_padding=0, groups=1, bias=True, coord_conv=True):
    if coord_conv:
        return CoordConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                                    padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                                  padding=padding, output_padding=output_padding, dilation=dilation, bias=bias)
    

    
class BasicBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1, coord_conv=True):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, coord_conv=coord_conv)
        self.conv2 = Conv2d(out_filters, out_filters, kernel_size=3, stride=stride, padding=1, coord_conv=coord_conv)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
    
        out += residual
        
        return out
    
class Encoder(nn.Module):
    ''' 
    '''
    def __init__(self,coord_conv=True):
   
        super(Encoder, self).__init__()
        
        self.conv1 = Conv2d(15, 16, kernel_size=7, stride=1, padding=3, coord_conv=coord_conv)
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2, padding=1, coord_conv=coord_conv)
        self.conv3 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1, coord_conv=coord_conv)
        self.conv4 = Conv2d(64, 128, kernel_size=4, stride=2, padding=1, coord_conv=coord_conv)   
        self.conv5 = Conv2d(128, 256, kernel_size=4, stride=2, padding=1, coord_conv=coord_conv)
        self.conv6 = Conv2d(256, 512, kernel_size=4, stride=2, padding=1, coord_conv=coord_conv)
        
        layers=[]
        for i in range(4):
            layers.append(BasicBlock(512, 512))
        self.layer = nn.Sequential(*layers)
            
    def forward(self, x):
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        
        out = self.layer(out)

        return out #(4x4x512)
    
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    #style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    style_mean = torch.mean(style_feat)
    style_std = torch.std(style_feat)
    
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)   

class AdaIN_BasicBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1, coord_conv=True):
        super(AdaIN_BasicBlock, self).__init__()
        
        self.conv1 = Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, coord_conv=coord_conv)
        
        self.conv2 = Conv2d(out_filters, out_filters, kernel_size=3, stride=stride, padding=1, coord_conv=coord_conv)

    def forward(self, x,pose):
        residual = x
        out = self.conv1(x)
        out = adaptive_instance_normalization(out,pose)
        out = F.relu(out)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out,pose)
    
        out += residual
        
        return out

class VIGAN_Decoder(nn.Module):
    def __init__(self):
        super(VIGAN_Decoder, self).__init__()
        self.MD_fc1 = nn.Linear(12,16)
        self.MD_fc2 = nn.Linear(16,16)
        self.MD_fc3 = nn.Linear(16,64)
        
# =============================================================================
#         layers=[]
#         for i in range(4):
#             layers.append(AdaIN_BasicBlock(576, 576))
#         self.layer = nn.Sequential(*layers)
# =============================================================================
        
        self.resblock1 = AdaIN_BasicBlock(576, 576)
        self.resblock2 = AdaIN_BasicBlock(576, 576)
        self.resblock3 = AdaIN_BasicBlock(576, 576)
        self.resblock4 = AdaIN_BasicBlock(576, 576)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = Conv2d(576, 288, kernel_size=5, stride=1, padding=2, coord_conv=True)
        self.LN1 = nn.LayerNorm((288,8,8))
        self.conv2 = Conv2d(288, 144, kernel_size=5, stride=1, padding=2, coord_conv=True)
        self.LN2  = nn.LayerNorm((144,16,16))
        self.conv3 = Conv2d(144, 72, kernel_size=5, stride=1, padding=2, coord_conv=True)
        self.LN3 = nn.LayerNorm((72,32,32))
        self.conv4 = Conv2d(72, 36, kernel_size=5, stride=1, padding=2, coord_conv=True)
        self.LN4 = nn.LayerNorm((36,64,64))
        self.conv5 = Conv2d(36, 18, kernel_size=5, stride=1, padding=2, coord_conv=True)
        self.LN5 = nn.LayerNorm((18,128,128))
        
        self.conv6 = Conv2d(18, 3, kernel_size=7, stride=1, padding=3, coord_conv=True)
        self.tanh = nn.Tanh()
        
        
    def forward(self, x, pose):
        
        pose = F.relu(self.MD_fc1(pose))
        pose = F.relu(self.MD_fc2(pose))
        pose = self.MD_fc3(pose)
        
        pose1 = pose[:,:,None,None]
        pose1 = pose1.repeat(1,1,4,4)
        x = torch.cat((x,pose1),dim=1)
        
        #x = self.layer(x,pose)
        x = self.resblock1.forward(x,pose)
        x = self.resblock2.forward(x,pose)
        x = self.resblock3.forward(x,pose)
        x = self.resblock4.forward(x,pose)
        x = F.relu(self.LN1(self.conv1(self.upsample(x))))
        x = F.relu(self.LN2(self.conv2(self.upsample(x))))
        x = F.relu(self.LN3(self.conv3(self.upsample(x))))
        x = F.relu(self.LN4(self.conv4(self.upsample(x))))
        x1 = F.relu(self.LN5(self.conv5(self.upsample(x))))
        
        out = self.tanh(self.conv6(x1))
        return out
        
    
class VIGAN_Discriminator(nn.Module):
    def __init__(self):
        super(VIGAN_Discriminator, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv3 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv4 = Conv2d(64, 128, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv5 = Conv2d(128, 256, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv6 = Conv2d(256, 1, kernel_size=1, stride=1, padding=0, coord_conv=False)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        out = self.conv6(x)
    
        return out

class VIGANPoseDiscriminator(nn.Module):
    def __init__(self):
        super(VIGANPoseDiscriminator, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv2 = Conv2d(16, 32, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv3 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv4 = Conv2d(64, 128, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv5 = Conv2d(128, 256, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv6 = Conv2d(256, 512, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv7 = Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, coord_conv=True)
        self.conv8 = Conv2d(1024, 12, kernel_size=1, stride=1, padding=0, coord_conv=False)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        out = self.conv8(x)
    
        return out
    
    
class UpBasicBlock(nn.Module):
    def __init__(self, in_filters, out_filters,kernel_size=4, stride=2, upsample=None,
                 dilation=1, norm_type='instance', nonlinear_type='LeakyReLU', coord_conv=True):
        super(UpBasicBlock, self).__init__()
        self.conv1 = ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=1,
                            dilation=dilation, coord_conv=coord_conv)
        self.norm1 = get_norm_layer(out_filters, norm_type)
        self.relu = get_nonlinear_layer(nonlinear_type)
        self.conv2 = ConvTranspose2d(out_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=1,
                            dilation=dilation, coord_conv=coord_conv)
        self.norm2 = get_norm_layer(out_filters, norm_type)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        
        out1 = self.conv1(x)
        out2 = self.norm1(out1)
        out3 = self.relu(out2)
        out4 = self.conv2(out3)
        out5 = self.norm2(out4)

        residual = self.upsample(x)
        
        out5 += residual
        out = self.relu(out5)
        return out



class Decoder(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, norm_layer='instance', coord_conv=True):
        '''
        :param block: Either basic block or up sample block
        :param [layers]: How many ResNet layers to use at each level
        :param zero_init_residual: Whether to zero intialize the residual layers
        :param norm: Which norm to use. Default is Batchnorm.
        '''
        super(Decoder, self).__init__()
        self.in_filters = 512
        self.dilation = 1
        self.coord_conv = coord_conv
        n_filters = 512
            
        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,1024)
        self.fc3 = nn.Linear(1024,2048)
        self.fc4 = nn.Linear(2048,4096)       
        
        self.relu = get_nonlinear_layer('LeakyReLU')
        self.upsampling1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(16,8,kernel_size=1,stride=1,padding=0))
        self.upsampling2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(8,4,kernel_size=1,stride=1,padding=0))
        self.upsampling3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(4,3,kernel_size=1,stride=1,padding=0))
        #self.layer1 = self._make_layer(block,n_filters,int(n_filters/2), layers[1], stride=2) #(1,1,512) -> (4,4,256)
        #self.layer2 = self._make_layer(block,int(n_filters/2),int(n_filters/4), layers[1], stride=2) #(4,4,256) -> (16,16,128)
        #self.layer3 = self._make_layer(block,int(n_filters/4),int(n_filters/8), layers[1], stride=2) #(16,16,128) -> (64,64,64)
        #(1,1,512) -> (4,4,256)

        self.conv1_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1))
        self.norm1 = get_norm_layer(8, norm_layer)
        self.conv1_2 = Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True, coord_conv=True)
        #(16,16,16) -> (32,32,8)

        self.conv2_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(8,4,kernel_size=3,stride=1,padding=1))
        self.norm2 = get_norm_layer(4, norm_layer)
        self.conv2_2 = Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True, coord_conv=True)
        #(32,32,8) -> (64,64,4)

        self.conv3_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(4,3,kernel_size=3,stride=1,padding=1))
        self.norm3 = get_norm_layer(3, norm_layer)
        self.conv3_2 = Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True, coord_conv=True)
        #layer4 (64,64,4) -> (64,64,3)
        self.conv4_1 = Conv2d(3, 3, kernel_size=3, stride=1, padding=1, coord_conv=True)
        self.norm4 = get_norm_layer(3, norm_layer)
        self.conv4_2 = Conv2d(3, 3, kernel_size=3, stride=1, padding=1, coord_conv=True)
        #last conv
        self.upconv5 = Conv2d(3, 3, kernel_size=3, stride=1, padding=1, coord_conv=True)    
        self.tanh = nn.Tanh()
        
        # Initialize layers
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            #elif isinstance(m, BasicBlock):
                #nn.init.constant_(m.norm2.weight, 0)    
                
        self.MD_fc1 = nn.Linear(12,128)
        self.MD_fc2 = nn.Linear(128,256)
        self.MD_fc3 = nn.Linear(256,512)
        self.MD_fc4 = nn.Linear(512,128)
        
        
    def _make_layer(self, block, in_filters, out_filters, blocks, kernel_size=4, stride=2):
        upsample = None
        if stride != 1 or in_filters != out_filters:
            upsample = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear'),
                                      nn.Conv2d(in_filters,out_filters,kernel_size=1,stride=1,padding=0))

        layers = []
        layers.append(block(in_filters, out_filters,kernel_size, stride, upsample, coord_conv=self.coord_conv))
        in_filters = out_filters 
        for _ in range(1, blocks):
            layers.append(block(in_filters, out_filters,kernel_size, stride, upsample, coord_conv=self.coord_conv))
        return nn.Sequential(*layers)

    def forward(self, x, pose):
        
        pose = F.relu(self.MD_fc1(pose))
        pose = F.relu(self.MD_fc2(pose))
        pose = F.relu(self.MD_fc3(pose))
        pose = self.MD_fc4(pose)        
        
        x = torch.cat((x,pose),1)
        
        #x += torch.zeros_like(x).normal_(mean=0.0, std=0.2).to(x.device)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  #4096
        
        x = torch.reshape(x, (2, 16,16,16))
        
        #x1 = self.layer1(x) #(1,1,512) -> (4,4,256)
        #x2 = self.layer2(x1) #(4,4,256) -> (16,16,128)
        #x3 = self.layer3(x2) #(16,16,128) -> (64,64,64)
        #layer1
        x1 = self.conv1_1(x)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.norm1(x1)
        x1 += self.upsampling1(x) #cat
        x1 = F.relu(x1)
        #layer2
        x2 = self.conv2_1(x1) 
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x2 = self.conv2_2(x2)
        x2 = self.norm4(x2)
        x2 += self.upsampling2(x1)
        x2 = F.relu(x2)
        #layer3
        x3 = self.conv3_1(x2) 
        x3 = self.norm2(x3)
        x3 = F.relu(x3)
        x3 = self.conv3_2(x3)
        x3 = self.norm4(x3)
        x3 += self.upsampling3(x2)
        x3 = F.relu(x3)
        #layer4
        x4 = self.conv4_1(x3) 
        x4 = self.norm4(x4)
        x4 = self.relu(x4)
        x4 = self.conv4_2(x4)
        x4 = self.norm4(x4)
        x4 += x3
        x4 = self.relu(x4)
        
        out = self.upconv5(x4) 
        out = self.tanh(out)

        return out



        

class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, sigma=0.2, num_intermediate_layers=0):
        super(PatchImageDiscriminator, self).__init__()

        self.noise_layer = Noise()
        self.sigma = sigma

        layers = []
        layers.append(self.noise_layer)
        layers.append(nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(self.noise_layer)
        layers.append(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for layer_idx in range(num_intermediate_layers):
            layers.append(self.noise_layer)
            layers.append(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ndf * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(self.noise_layer)
        layers.append(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        layers.append(nn.BatchNorm2d(ndf * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(self.noise_layer)
        layers.append(nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False))
        layers.append(nn.Conv2d(1, 1, 8, 1, bias=False))
        self.main = nn.Sequential(*layers)

    def set_sigma(self, sigma):
        self.noise_layer.set_sigma(sigma)

    def forward(self, input):
        h = self.main(input).squeeze()
        return h[:,None]
    
class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()
        self.std = 0.2

    def set_sigma(self, std):
        self.std = std

    def forward(self, x):
    	return x + torch.zeros_like(x).normal_(mean=0.0, std=self.std).to(x.device)

class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True,
                 path=None):
        super(PerceptualVGG19, self).__init__()
        if path != '' and path is not None:
            print('Loading pretrained model')
            model = models.vgg19(pretrained=False)
            model.load_state_dict(torch.load(path))
        else:
            model = models.vgg19(pretrained=True)
        model.float()
        model.eval()

        self.model = model
        self.feature_layers = feature_layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None

        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None

        self.use_normalization = use_normalization

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if not self.use_normalization:
            return x

        if self.mean_tensor is None:
            self.mean_tensor = self.mean.view(1, 3, 1, 1).expand(x.size())
            self.std_tensor = self.std.view(1, 3, 1, 1).expand(x.size())

        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        features = []

        h = x
        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)
        return torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)
