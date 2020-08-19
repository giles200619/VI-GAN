# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:04:01 2020

@author: giles
"""

from __future__ import print_function
import torch
import torch.nn as nn
import os
import random
import argparse
import matplotlib.pyplot as plt
from network_VIGAN import *
from utils import *
from resnet import resnet18
from torch.utils.tensorboard import SummaryWriter
import warnings
#from dataloader import ImagePairDataset
warnings.filterwarnings(action='once')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train',  default=1 ,type=int, help='1 for train, 2 for continue training')
parser.add_argument('--batch_size', dest='batch_size',  default=2, type=int, help='batch size')
parser.add_argument('--epoch', dest='epoch',  default=50 ,type=int, help='epoch')
parser.add_argument('--train_dir', dest='train_dir',  default='./chairs/chair_train' , help='training data dir')
parser.add_argument('--checkpoint', dest='checkpoint',  default='./checkpoint/2000.pth' , help='checkpoint path')
parser.add_argument('--save_frequency', dest='save_frequency',  default=1000 ,type=int, help='save model every # of iteration')
parser.add_argument('--print_frequency', dest='print_frequency',  default=10,type=int, help='print loss every # of iteration')
parser.add_argument('--multi_GPU', dest='multi_GPU',  default=True,type=bool, help='use multi GPU or not')
parser.add_argument('--WGAN', dest='WGAN',  default=False,type=bool, help='use WGAN-GP or not')
#parser.add_argument('--checkpoint_pose', dest='checkpoint_pose',  default='C:/Users/giles/work/CVPR_nvs/checkpoint_pose/14_5000.pth' , help='checkpoint path')
parser.add_argument('--lambda_cyc', type=float, default=1.0, help='weight of cycle loss')
parser.add_argument('--lambda_recons', type=float, default=1.0, help='weight of 3D reconstruction loss ')
parser.add_argument('--lambda_pixel', type=float, default=1.0, help='weigth of pixel level loss')
parser.add_argument('--lambda_VGG', type=float, default=0.0, help='weight of perceptual loss')
parser.add_argument('--lambda_cyc_VGG', type=float, default=0.0, help='weight of cycle perceptual loss')
parser.add_argument('--lambda_VI', type=float, default=0.0, help='weight of view independent loss')
parser.add_argument('--lambda_G', type=float, default=0.0, help='weigth of GAN loss')
parser.add_argument('--lambda_pose', type=float, default=0.0, help='weigth of pose prediction loss')   
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
    
    if args.multi_GPU:
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
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.0001, betas=(0.5, 0.999))    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_pose = torch.optim.Adam(poseDiscriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    writer = SummaryWriter(log_dir='log_'+os.path.basename(args.train_dir))
    
    #data = ImagePairDataset(args.train_dir)
    #dataloader = torch.utils.data.DataLoader(data,batch_size=args.batch_size, shuffle=True)
    
    #train or continue
    if args.train == 1:
        
        name_pair = dataNamePair(args.train_dir)
        random.shuffle(name_pair)
        iterations = int(len(name_pair)/args.batch_size)
        
        print("Training iterations:",iterations)
        print("Epoch:", args.epoch)
        print("Start Training...")
        epoch = 0
        it = 0
        
    if args.train == 2:
        print("Continue Training...")
        name_pair = dataNamePair(args.train_dir)
        random.shuffle(name_pair)
        iterations = int(len(name_pair)/args.batch_size)
        if not (os.path.isfile(args.checkpoint) or os.path.splitext(args.checkpoint)[1] == '.pth'):
            raise OSError('Invalid checkpoint.')
        checkpoint = torch.load(args.checkpoint)
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        discriminator.load_state_dict(checkpoint['discriminator_state'])
        poseDiscriminator.load_state_dict(checkpoint['pose_state'])
        optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder'])
        optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        optimizer_pose.load_state_dict(checkpoint['optimizer_pose'])
        
        epoch =  checkpoint['epoch']
        it = checkpoint['iteration']
        it+=1
        print("Iteration:",epoch*iterations+it)
        
    

    for e in range(epoch,args.epoch):
        #scheduler.step()
        for i in range(it,iterations):
            
            optimizer_discriminator.zero_grad()
            optimizer_pose.zero_grad()
            
            filename_list = name_pair[i*args.batch_size:(i+1)*args.batch_size]
            img1, img2, poseA, poseB = imgPreprocess(filename_list,args)
            groundtruth_pose = groundTruthTensor(filename_list)
            input1 = catImgPose(img1,poseA)
            
            input1 = input1.to(device)
            img1 = img1.to(device)
            img2 = img2.to(device)
            poseA = poseA.to(device)
            poseB = poseB.to(device)
            groundtruth_pose = groundtruth_pose.to(device)
                            
            FA = encoder.forward(input1)
            imgA2B = decoder.forward(FA, poseB)
            
            l1 = nn.L1Loss()
            l2 = nn.MSELoss()
            #train discriminator
            for p in poseDiscriminator.parameters():  
                p.requires_grad = True 
            for p in discriminator.parameters():  
                p.requires_grad = True 
            for p in encoder.parameters():
                p.requires_grad = False
            for p in decoder.parameters():
                p.requires_grad = False
            
            #WGAN-GP
            if args.WGAN:
                loss_D_real = torch.mean(discriminator.forward(img2))
                loss_D_fake = torch.mean(discriminator.forward(imgA2B))
                gradient_penalty = calc_gradient_penalty(discriminator, img2.data, imgA2B.data,args,device)
                d_loss = loss_D_fake - loss_D_real + gradient_penalty
            else: #normal gan loss
                loss_D_real = gan(discriminator.forward(img2), True).mean()
                loss_D_fake = gan(discriminator.forward(imgA2B), False).mean()
                d_loss = loss_D_real + loss_D_fake
            d_loss.backward()
            #pose loss
            pre_pose = poseDiscriminator.forward(img2)
            pre_pose = pre_pose.view(pre_pose.size()[0],12)
            loss_D_P = l2(pre_pose, poseB)
            
            loss_D_P.backward()
            
            optimizer_discriminator.step()
            optimizer_pose.step()
            
            #train generator
            for p in poseDiscriminator.parameters():  
                p.requires_grad = False 
            for p in discriminator.parameters():  
                p.requires_grad = False
            for p in encoder.parameters():
                p.requires_grad = True
            for p in decoder.parameters():
                p.requires_grad = True
            
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            
            FA = encoder.forward(input1)
            imgA2B = decoder.forward(FA, poseB)
            
            recons_img1 = decoder.forward(FA, poseA)
            
            input2 = catImgPose(imgA2B,poseB)
            FB = encoder.forward(input2)
            imgB2A = decoder.forward(FB, poseA)
            
            #View-independent = FA-FB
            VI_loss = l1(FA, FB)
            #pixel-level = imgA2B - img2
            PL_loss = l1(imgA2B, img2)
            #perceptive = (VGG(imgA2B)-VGG(img2))^2
            upsample224 = nn.Upsample(size=224, mode='bilinear')
            perceptive_loss = VGG(upsample224(img2),upsample224(imgA2B)).mean()
            #cycle = img1-imgB2A
            cyc_loss = l1(img1, imgB2A)
            #cycle-perceptive = (VGG(imgB2A)-VGG(img1))^2
            cyc_per_loss = VGG(upsample224(img1), upsample224(imgB2A)).mean()
            #reconstruction = img1 - recons_img1
            reconstruction_loss = l1(img1, recons_img1)
            #WGAN_GP
            if args.WGAN:
                g_loss = -torch.mean(discriminator(imgA2B))
            else: #normal GAN
                g_loss = gan(discriminator(imgA2B), True).mean()
            #SiamesePoseNet
            #upsample128 = nn.Upsample(size=100, mode='bilinear')
            #output_pose = posenet.forward(upsample128(img1),upsample128(imgA2B))
            #pose_loss = l1(output_pose, groundtruth_pose)
            #pose_loss = ((poseDiscriminator.forward(imgA2B)[:2] - poseB) ** 2).mean()   
            pre_pose = poseDiscriminator.forward(imgA2B)
            pre_pose = pre_pose.view(pre_pose.size()[0],12)
            pose_loss = l2(pre_pose, poseB)
                
            loss = VI_loss*args.lambda_VI + PL_loss*args.lambda_pixel + perceptive_loss*args.lambda_VGG + \
                    cyc_loss*args.lambda_cyc + cyc_per_loss*args.lambda_cyc_VGG + reconstruction_loss*args.lambda_recons + \
                    g_loss*args.lambda_G + pose_loss*args.lambda_pose  
            #loss = PL_loss + cyc_loss + reconstruction_loss            
            loss.backward()

            optimizer_encoder.step()
            optimizer_decoder.step()
            
            if (e*iterations+i)%args.print_frequency == args.print_frequency-1:
                print(loss.item(),'iter:',i+1,f'[{e}/{args.epoch}]')
                print("d_loss:","{:.2f}".format(d_loss.item()),"g_loss:","{:.2f}".format(g_loss.item()),"d_Pose_loss","{:.2f}".format(loss_D_P.item()),
                      "View-Ind:","{:.2f}".format(VI_loss.item()),"Pixel:","{:.2f}".format(PL_loss.item()),
                      "percep:","{:.2f}".format(perceptive_loss.item()), "cyc:","{:.2f}".format(cyc_loss.item()),
                      "cyc_per:","{:.2f}".format(cyc_per_loss.item()),"recons:","{:.2f}".format(reconstruction_loss.item()),
                      "pose:","{:.2f}".format(pose_loss.item()))
                
                writer.add_scalar('Loss/train', loss.item(), (e*iterations+i+1))
                writer.add_scalar('Discriminator Loss', d_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Generator Loss', g_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Poss Discriminator Loss', loss_D_P.item(), (e*iterations+i+1))
                writer.add_scalar('View-independent Loss', VI_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Pixel-level Loss', PL_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Perceptive Loss', perceptive_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Cycle Loss', cyc_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Cycle-perceptive Loss', cyc_per_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Reconstruction Loss', reconstruction_loss.item(), (e*iterations+i+1))
                writer.add_scalar('Pose Loss', pose_loss.item(), (e*iterations+i+1))
                
            
            if (e*iterations+i)%args.save_frequency == args.save_frequency-1:
                torch.save({
                            'epoch': e,
                            'iteration': i,
                            'encoder_state': encoder.state_dict(),'decoder_state': decoder.state_dict(),'discriminator_state': discriminator.state_dict(),
                            'pose_state': poseDiscriminator.state_dict(), 'optimizer_pose': optimizer_pose.state_dict(),
                            'optimizer_encoder': optimizer_encoder.state_dict(),'optimizer_decoder': optimizer_decoder.state_dict(),
                            'optimizer_discriminator': optimizer_discriminator.state_dict(),                            
                            }, os.path.dirname(args.checkpoint)+f'/{e*iterations+i+1}.pth')
                FA = encoder.forward(input1)
                imgA2B = decoder.forward(FA, poseB)
                recons_img1 = decoder.forward(FA, poseA)
                input2 = catImgPose(imgA2B,poseB)
                FB = encoder.forward(input2)
                imgB2A = decoder.forward(FB, poseA)
                
                fig, axs = plt.subplots(1, 5)
                axs[0].imshow(tensor2im(img1))
                axs[0].axis('off')
                axs[0].title.set_text('Img 1')
                axs[1].imshow(tensor2im(imgB2A))
                axs[1].axis('off')
                axs[1].title.set_text('Img B2A')
                axs[2].imshow(tensor2im(recons_img1))
                axs[2].axis('off')
                axs[2].title.set_text('recons img1')
                axs[3].imshow(tensor2im(img2))
                axs[3].axis('off')
                axs[3].title.set_text('Img 2')
                axs[4].imshow(tensor2im(imgA2B))
                axs[4].axis('off')
                axs[4].title.set_text('Img A2B')
                writer.add_figure('Testing results', fig, global_step=e*iterations+i+1)
            
        it = 0
    print('Finished training.')
    

    