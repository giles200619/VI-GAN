# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:04:01 2020

@author: giles
"""

from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from PIL import Image
from network_VIGAN import *
from utils import *
import warnings
from pytorch_ssim import ssim
warnings.filterwarnings(action='once')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train',  default=0 ,type=int, help='0 for testing')
parser.add_argument('--print_frequency', dest='print_frequency',  default=10,type=int, help='print loss every # of iteration')
parser.add_argument('--WGAN', dest='WGAN',  default=False,type=bool, help='use WGAN-GP or not')
parser.add_argument('--batch_size', dest='batch_size',  default=1, type=int, help='batch size')
parser.add_argument('--test_dir', dest='test_dir',  default='./chairs/chair_test' , help='testing data dir')
parser.add_argument('--checkpoint', dest='checkpoint',  default='./chair_68000.pth' , help='checkpoint path')
args = parser.parse_args()
    

if __name__ == "__main__":    
    
    print("cuda avail:",torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    
    encoder = Encoder()
    #encoder = resnet18(pretrained=True)
    #encoder = nn.Sequential(*list(encoder.children())[:-1],nn.Flatten(),nn.Linear(in_features=512, out_features=128, bias=True))
    #encoder[0] = nn.Conv2d(15, 64, kernel_size=7, stride=2, padding=3,bias=False)
    encoder = encoder.float()
    #encoder.forward(torch.randn(2,15,128,128).type(torch.double))
    #decoder = Decoder(UpBasicBlock, [1,1,1,1], zero_init_residual=False, norm_layer='instance', coord_conv=True)
    decoder = VIGAN_Decoder()
    decoder = decoder.float()
    #decoder.forward(torch.randn(2,128),torch.randn(2,12))
    
    #discriminator = PatchImageDiscriminator(3)
    discriminator = VIGAN_Discriminator()
    discriminator = discriminator.float()
    
    gan = GANLoss()
    VGG = PerceptualLoss()
    
    poseDiscriminator = VIGANPoseDiscriminator()
    poseDiscriminator.float()
    

    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    discriminator = nn.DataParallel(discriminator)
    poseDiscriminator = nn.DataParallel(poseDiscriminator)
    VGG = nn.DataParallel(VGG)
    gan = nn.DataParallel(gan)
    
    encoder.to(device)
    decoder.to(device)
    VGG = VGG.to(device)
    discriminator.to(device)
    poseDiscriminator.to(device)
    gan = gan.to(device)
# =============================================================================
#     posenet = SiamesePoseNet()
#     posenet = posenet.float()
#     posenet.to(device)
#     checkpoint = torch.load(args.checkpoint_pose)
#     posenet.load_state_dict(checkpoint['model_state_dict'])
# =============================================================================
    #pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    #writer = SummaryWriter(lorr=args.log_dir)
    
    name_pair = dataNamePair(args.test_dir)
    iterations = int(len(name_pair)/args.batch_size)
    
    
    if not (os.path.isfile(args.checkpoint) or os.path.splitext(args.checkpoint)[1] == '.pth'):
        raise OSError('Invalid checkpoint.')
    checkpoint = torch.load(args.checkpoint)
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    discriminator.load_state_dict(checkpoint['discriminator_state'])
    poseDiscriminator.load_state_dict(checkpoint['pose_state'])
#
#
    l1_loss_fn = nn.L1Loss()
    l1_results = np.empty(iterations)
    ssim_results = np.empty(iterations)
    
    if not os.path.isdir("./ouptut"):
        os.mkdir("./ouptut")
        os.mkdir("./ouptut/target")
        os.mkdir("./ouptut/predict")
    for i in range(iterations):
        filename_list = name_pair[i*args.batch_size:(i+1)*args.batch_size]
        img1, img2, poseA, poseB = imgPreprocess(filename_list,args)
        #groundtruth_pose = groundTruthTensor(filename_list)
        input1 = catImgPose(img1,poseA)
        
        input1 = input1.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        poseA = poseA.to(device)
        poseB = poseB.to(device)
        #groundtruth_pose = groundtruth_pose.to(device)
                        
        FA = encoder.forward(input1)
        imgA2B = decoder.forward(FA, poseB)
        
        name = filename_list[0][0]+"_to_"+filename_list[0][1].split("_")[1]+"_"+filename_list[0][1].split("_")[2]+"_"
        pre_path = "./ouptut/predict/"+name+"predict.png"
        gt_path = "./ouptut/target/"+name+"target.png"
        
        
        Image.fromarray(tensor2im(imgA2B).astype('uint8'), 'RGB').save(pre_path)
        Image.fromarray(tensor2im(img2).astype('uint8'), 'RGB').save(gt_path)
        
        imgA2B = torch.tensor(tensor2im(imgA2B)/255).float().permute(2,0,1).unsqueeze(0)
        img2 = torch.tensor(tensor2im(img2)/255).float().permute(2,0,1).unsqueeze(0)

        l1_loss = l1_loss_fn(imgA2B, img2)
        l1_results[i] = l1_loss
        
        ssim_loss = ssim(imgA2B, img2)
        ssim_results[i] = ssim_loss
        
        if i % 100 == 0:
            print("Processing..."+str(i))
    print(f"L1 loss mean: {l1_results.mean()}, std: {l1_results.std()}")
    print(f"SSIM loss mean: {ssim_results.mean()}, std: {ssim_results.std()}")
        
        #Image.fromarray(source_img.astype('uint8'), 'RGB').save("aaa.png")
        
# =============================================================================
#         l1 = nn.L1Loss()
#         l2 = nn.MSELoss()
#         
#         #WGAN-GP
#         if args.WGAN:
#             loss_D_real = torch.mean(discriminator.forward(img2))
#             loss_D_fake = torch.mean(discriminator.forward(imgA2B))
#             gradient_penalty = calc_gradient_penalty(discriminator, img2.data, imgA2B.data)
#             d_loss = loss_D_fake - loss_D_real + gradient_penalty
#         else: #normal gan loss
#             loss_D_real = gan(discriminator.forward(img2), True).mean()
#             loss_D_fake = gan(discriminator.forward(imgA2B), False).mean()
#             d_loss = loss_D_real + loss_D_fake
# 
#         #pose loss
#         pre_pose = poseDiscriminator.forward(img2)
#         pre_pose = pre_pose.view(pre_pose.size()[0],12)
#         loss_D_P = l2(pre_pose, poseB)
#         
#         
#         FA = encoder.forward(input1)
#         imgA2B = decoder.forward(FA, poseB)
#         
#         recons_img1 = decoder.forward(FA, poseA)
#         
#         input2 = catImgPose(imgA2B,poseB)
#         FB = encoder.forward(input2)
#         imgB2A = decoder.forward(FB, poseA)
#         
#         #View-independent = FA-FB
#         VI_loss = l1(FA, FB)
#         #pixel-level = imgA2B - img2
#         PL_loss = l1(imgA2B, img2)
#         #perceptive = (VGG(imgA2B)-VGG(img2))^2
#         upsample224 = nn.Upsample(size=224, mode='bilinear')
#         perceptive_loss = VGG(upsample224(img2),upsample224(imgA2B)).mean()
#         #cycle = img1-imgB2A
#         cyc_loss = l1(img1, imgB2A)
#         #cycle-perceptive = (VGG(imgB2A)-VGG(img1))^2
#         cyc_per_loss = VGG(upsample224(img1), upsample224(imgB2A)).mean()
#         #reconstruction = img1 - recons_img1
#         reconstruction_loss = l1(img1, recons_img1)
#         #WGAN_GP
#         if args.WGAN:
#             g_loss = -torch.mean(discriminator(imgA2B))
#         else: #normal GAN
#             g_loss = gan(discriminator(imgA2B), True).mean()
#         #SiamesePoseNet
#         #upsample128 = nn.Upsample(size=100, mode='bilinear')
#         #output_pose = posenet.forward(upsample128(img1),upsample128(imgA2B))
#         #pose_loss = l1(output_pose, groundtruth_pose)
#         #pose_loss = ((poseDiscriminator.forward(imgA2B)[:2] - poseB) ** 2).mean()   
#         pre_pose = poseDiscriminator.forward(imgA2B)
#         pre_pose = pre_pose.view(pre_pose.size()[0],12)
#         pose_loss = l2(pre_pose, poseB)
#         
#         #loss = VI_loss + PL_loss + perceptive_loss + cyc_loss + cyc_per_loss + reconstruction_loss + g_loss + pose_loss  
#         loss = PL_loss + cyc_loss + reconstruction_loss
#         
#         
#         if (iterations+i)%args.print_frequency == args.print_frequency-1:
#             print(loss.item(),'iter:',i+1)
#             print("d_loss:","{:.2f}".format(d_loss.item()),"g_loss:","{:.2f}".format(g_loss.item()),"d_Pose_loss","{:.2f}".format(loss_D_P.item()),
#                   "View-Ind:","{:.2f}".format(VI_loss.item()),"Pixel:","{:.2f}".format(PL_loss.item()),
#                   "percep:","{:.2f}".format(perceptive_loss.item()), "cyc:","{:.2f}".format(cyc_loss.item()),
#                   "cyc_per:","{:.2f}".format(cyc_per_loss.item()),"recons:","{:.2f}".format(reconstruction_loss.item()),
#                   "pose:","{:.2f}".format(pose_loss.item()))
#             
#             writer.add_scalar('Loss', loss.item(), (i+1))
#             writer.add_scalar('Discriminator Loss', d_loss.item(), (i+1))
#             writer.add_scalar('Generator Loss', g_loss.item(), (i+1))
#             writer.add_scalar('Poss Discriminator Loss', loss_D_P.item(), (i+1))
#             writer.add_scalar('View-independent Loss', VI_loss.item(), (i+1))
#             writer.add_scalar('Pixel-level Loss', PL_loss.item(), (i+1))
#             writer.add_scalar('Perceptive Loss', perceptive_loss.item(), (i+1))
#             writer.add_scalar('Cycle Loss', cyc_loss.item(), (i+1))
#             writer.add_scalar('Cycle-perceptive Loss', cyc_per_loss.item(), (i+1))
#             writer.add_scalar('Reconstruction Loss', reconstruction_loss.item(), (i+1))
#             writer.add_scalar('Pose Loss', pose_loss.item(), (i+1))
#             writer.add_scalar('FA', (FA-pre_FA).mean(), (i+1))
#             
#             pre_FA = FA
#             fig, axs = plt.subplots(1, 5)
#             axs[0].imshow(tensor2im(img1))
#             axs[0].axis('off')
#             axs[0].title.set_text('Img 1')
#             axs[1].imshow(tensor2im(imgB2A))
#             axs[1].axis('off')
#             axs[1].title.set_text('Img B2A')
#             axs[2].imshow(tensor2im(recons_img1))
#             axs[2].axis('off')
#             axs[2].title.set_text('recons img1')
#             axs[3].imshow(tensor2im(img2))
#             axs[3].axis('off')
#             axs[3].title.set_text('Img 2')
#             axs[4].imshow(tensor2im(imgA2B))
#             axs[4].axis('off')
#             axs[4].title.set_text('Img A2B')
#             writer.add_figure('Testing results', fig, global_step=i+1)
# =============================================================================
            


    