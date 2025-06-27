import os
import cv2
import numpy as np;
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
# import segmentation_models_pytorch as smp
# import albumentations as album

from typing import Tuple, List
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torchvision.ops import DropBlock2D, DropBlock3D
from collections import OrderedDict
from PIL import Image
import re, math, random
import pickle as p

from resnet import ResNet, BasicBlock

# import segmentation_models_pytorch.utils.metrics
    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, device=torch.device("cpu")):
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.device = device
        self.alpha = alpha.to(device)
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], loss_fn = F.binary_cross_entropy_with_logits, fn_reweight=False):
        if fn_reweight: 
            fn_wt = (targets > 1) + 1 
        
        targets = targets != 0
        targets = targets.to(torch.float32)
        bce_loss = loss_fn(inputs, targets, reduction="none") if loss_fn is F.binary_cross_entropy_with_logits else loss_fn(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-bce_loss)

        
        targets = targets.to(torch.int64)
        loss = (1 if loss_fn is not F.binary_cross_entropy_with_logits else  self.alpha[targets.view(targets.shape[0], -1)].reshape(targets.shape)) * (1-pt) ** self.gamma * bce_loss 
        if fn_reweight:
            fn_wt[fn_wt == 2] = 5
            loss *= fn_wt # fn are weighted 5 times more than regulars
        if mito_mask != []:
            #first modify loss_mask, neuron_mask is always on.
            loss_mask = loss_mask | mito_mask
            # factor = 1
            # loss = loss * (1 + (mito_mask * factor))#weight this a bit more. 
        if loss_mask != []: 
            #better way? TODO: get rid of this if statement
            if len(loss.shape) > len(loss_mask.shape): loss = loss * loss_mask.unsqueeze(-1)
            else: loss = loss * loss_mask # remove everything that is a neuron body, except ofc if the mito_mask was on. 
        return loss.mean() 

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], loss_fn=None):
        #get the probabilities:
        probs = nn.Sigmoid()(inputs)
        n1 = probs * targets
        inv_targets = 1 - targets

        d = probs + targets

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            # loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            inv_targets[loss_mask] = 0
            d *= (loss_mask == 0)
            d += (loss_mask *2) # eqv to d[loss_mask]=2

        n2 = (1-probs) * inv_targets
    
        # no need to use masks for balancing, coz of the loss fn here

        n1 = n1.view(n1.shape[0], -1)
        n2 = n2.view(n2.shape[0], -1)
        d =  d.view(d.shape[0], -1)

        return torch.mean(2 - (torch.sum(n1, dim=-1) + self.eps)/(torch.sum(d, dim=-1) + self.eps)
                           - (torch.sum(n2, dim=-1) + self.eps)/(torch.sum(2-d, dim=-1) + self.eps))

class TverskyFocal(nn.Module):
    def __init__(self, eps=1e-6):
        super(TverskyFocal, self).__init__()
        self.eps = eps
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], recall=0.7, precision=0.3):
        #get the probabilities:
        probs = nn.Sigmoid()(inputs)
        n1 = probs * targets
        inv_targets = 1 - targets

        d = probs + targets

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            # loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            inv_targets[loss_mask] = 0
            d *= (loss_mask == 0)
            d += (loss_mask *2) # eqv to d[loss_mask]=2

        n2 = (1-probs) * inv_targets

        f1 = probs * inv_targets
        f2 = (1-probs) * targets
    
        # no need to use masks for balancing, coz of the loss fn here

        n1 = n1.view(n1.shape[0], -1)
        n2 = n2.view(n2.shape[0], -1)
        d =  d.view(d.shape[0], -1)

        f1 = torch.sum(f1.view(n1.shape[0], -1))
        f2 = torch.sum(f2.view(n1.shape[0], -1))

        return torch.mean(2 - (torch.sum(n1, dim=-1) + self.eps)/(torch.sum(n1, dim=-1) + recall*f2 + precision*f1  + self.eps)
                           - (torch.sum(n2, dim=-1) + self.eps)/(torch.sum(n2, dim=-1) + precision*f2 + recall*f1 + self.eps))

class SSLoss(nn.Module):
    def __init__(self, eps=1e-6, lamda=0.05):
        super(SSLoss, self).__init__()
        self.eps = eps
        self.lamda = 0.05

    def forward(self, inputs, targets, loss_mask=[], mito_mask=[]):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)
        inv_targets = 1-targets
        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            inv_targets[loss_mask] = 0

        return torch.mean(self.lmda * torch.sum(torch.pow((targets - inputs), 2) * targets, dim=-1)/(torch.sum(targets, dim=-1) + self.eps) +
                           (1-self.lamda) * torch.sum(torch.pow((targets - inputs), 2) * inv_targets, dim=-1)/ (torch.sum(inv_targets, dim=-1) + self.eps))

class SpecialLoss(nn.Module):
    def __init__(self, bg_imp=2e-2):
        super(SpecialLoss, self).__init__()
        self.bg_imp = bg_imp
    
    def forward(self, predictions, targets, neuron_mask=[], mito_mask=[]):
        target_centers, targets, pad_mask = targets

        pad_tensors = torch.zeros((target_centers.shape[-2:])).to(dtype=torch.bool).to(pad_mask.device)
        pad_tensors[0, 0] = 1 # simply 
        target_centers[pad_mask] = pad_tensors # for now

        rows, cols = np.indices(target_centers.shape[-2:])
        rows, cols = torch.from_numpy(rows).to(predictions.device), torch.from_numpy(cols).to(predictions.device)
        #extend 
        rows, cols = rows.expand(predictions.size(0), targets.size(1), -1, -1), cols.expand(predictions.size(0),targets.size(1), -1, -1)
        center_rows, center_cols = (rows * target_centers).sum(dim = -1, keepdim=True).sum(dim=-1, keepdim=True), (cols * target_centers).sum(dim = -1, keepdim=True).sum(dim=-1, keepdim=True) 
        
        squared_dist = (rows - center_rows) ** 2 + (cols - center_cols) ** 2
        pixel_importance = (targets) * 1/((squared_dist)+1e-6)
        pixel_importance[pad_mask] = 0 # reset everything that was purely pad

        pixel_importance = pixel_importance.sum(dim=1)
        importance_coeff = (targets.sum(dim=1) == 0) * self.bg_imp
        pixel_importance += importance_coeff

        assert torch.count_nonzero(pixel_importance <0 ) == 0

        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets.sum(dim=1).to(dtype=torch.float32), reduction="none")
        bce_loss *= pixel_importance

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        if neuron_mask != []:
            if len(targets.shape) > len(neuron_mask.shape): neuron_mask.unsqueeze(-1)
            # neuron_mask = neuron_mask.view(neuron_mask.shape[0], -1)
            bce_loss *= neuron_mask

        return torch.mean(bce_loss)
        


class GenDLoss(nn.Module):
    def __init__(self):
        super(GenDLoss, self).__init__()
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], loss_fn=None, fn_reweight=None):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)

        inputs = torch.stack([inputs, 1-inputs], dim=-1)
        targets = torch.stack([targets, 1-targets], dim=-1)

        if mito_mask != []:
            loss_mask = loss_mask | mito_mask 
        
        if loss_mask != []:
            if len(targets.shape) > len(loss_mask.shape): loss_mask.unsqueeze(-1)
            loss_mask = loss_mask.view(loss_mask.shape[0], -1)
            targets *= loss_mask.unsqueeze(-1)
            inputs *= loss_mask.unsqueeze(-1)#0 them out in both masks

        weights = 1 / (torch.sum(torch.permute(targets, (0, 2, 1)), dim=-1).pow(2)+1e-6)
        targets, inputs = torch.permute(targets, (0, 2, 1)), torch.permute(inputs, (0, 2, 1))

        # print(torch.nansum(weights * torch.nansum(targets * inputs, dim=-1), dim=-1))
        # print(weights)

        return torch.nanmean(1 - 2 * torch.nansum(weights * torch.nansum(targets * inputs, dim=-1), dim=-1)/\
                          torch.nansum(weights * torch.nansum(targets + inputs, dim=-1), dim=-1))

class MultiGenDLoss(nn.Module):
    def __init__(self):
        super(MultiGenDLoss, self).__init__()
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], classes=3, **kwargs):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], targets.shape[1], -1), inputs.view(inputs.shape[0], targets.shape[1], -1)

        weights = 1 / (torch.sum(targets, dim=-1).pow(2)+1e-6)
        # print(weights.shape, torch.nansum(targets * inputs, dim=-1).shape)
        return torch.nanmean(1 - 2 * torch.nansum(weights * torch.nansum(targets * inputs, dim=-1))/\
                          torch.nansum(weights * torch.nansum(targets + inputs, dim=-1)))


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, three=False, spatial=False, residual=False, dropout=0):
        super(DoubleConv, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.projection_add = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.final = nn.Sequential(
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            self.dropout,
        )
        self.spatial=spatial
        if spatial: 
            self.spatial_sample = PyramidPooling(levels=[2, 2, 4, 4, 4, 4, 4, 4,4, 4], td=three)
        self.residual = residual

    def forward(self, x_in):
        x = self.double_conv(x_in)
        # if self.residual: x = x + self.projection_add(x_in)
        # x = self.final(x)
        con_shape = x.shape
        # if self.spatial: # Spatial pyramidal pooling
        #     x = self.spatial_sample(x)
        #     x = x.reshape(con_shape)
        return x
    
class DownBlock(nn.Module):
    """Double Convolution followed by Max Pooling"""
    def __init__(self, in_channels, out_channels, three=False, spatial=True, dropout=0, residual=False):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, three=three, dropout=dropout, residual=residual)
        self.spatial = spatial
        if spatial: 
            self.spatial_sample = PyramidPooling(levels=[2, 2, 4, 4, 4, 4, 4, 4,4, 4], td=three)
        self.down_sample = nn.MaxPool2d(2, stride=2) if not three else nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        # if self.spatial: 
        #     x = self.spatial_sample(skip_out)
        #     x = x.reshape(skip_out.shape)
        #     down_out = self.down_sample(x)
        # else:   
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Up Convolution (Upsampling followed by Double Convolution)"""
    def __init__(self, in_channels, out_channels, up_sample_mode, kernel_size=2, three=False, dropout=0, residual=False):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            if three: self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
                nn.Batchnorm3d(in_channels-out_channels),
                nn.ReLU())       
            else: self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
                nn.BatchNorm2d(in_channels-out_channels),
                nn.ReLU())
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True, three=three)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels, three=three, residual=residual)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)

class UNet(nn.Module):
    """UNet Architecture"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, residual=False, scale=False, spatial=False, dropout=0, classes=2):
        """Initialize the UNet model"""
        super(UNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three, spatial=False, residual=residual) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three, spatial=spatial, dropout=self.dropout, residual=residual) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256, spatial=spatial, dropout=self.dropout, residual=residual) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512, spatial=spatial, dropout=self.dropout, residual=residual) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024,spatial=spatial, dropout=self.dropout, residual=residual)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, residual=residual) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, residual=residual)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1 if classes == 2 else classes, kernel_size=1)
        self.attend = attend
        if scale:
            self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)    
        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        if self.three: x = x.squeeze(-3)   
        x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class ResUNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, dropout=0):
        """Initialize the UNet model"""
        super(ResUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.resnet = ResNet(BasicBlock, [1, 4, 6, 3, 3], norm_layer = nn.BatchNorm2d if not three else nn.BatchNorm3d, three=three)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, kernel_size=2) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode, kernel_size=2)

        #extra convs:
        self.m1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64,kernel_size=2, stride=2),
                nn.ReLU())
        self.m2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.ReLU())

        self.add_conv1 = DoubleConv(64, 64)
        self.add_conv2 = DoubleConv(64, 64)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        # if scale:
        #     self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


    def forward(self, x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out, skip2_out, skip3_out, skip4_out = self.resnet(x)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        # if self.three: 
        #     #attention_mode???
        #     skip1_out = torch.mean(skip1_out, dim=2)
            # skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.m1(x)
        x = self.add_conv1(x)
        x = self.m2(x)
        x = self.add_conv2(x)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class MemResUNet(nn.Module):
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, dropout=0):
        """Initialize the UNet model"""
        super(MemResUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        self.dropout=dropout

        # Downsampling Path
        self.resnet = ResNet(BasicBlock, [1, 4, 6, 3, 3], norm_layer = nn.BatchNorm2d if not three else nn.BatchNorm3d, three=three, membrane=True)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode, dropout=self.dropout, kernel_size=2) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode, dropout=self.dropout, kernel_size=2)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode, kernel_size=2)

        #extra convs:
        self.m1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64,kernel_size=2, stride=2),
                nn.ReLU())
        self.m2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                nn.ReLU())

        self.add_conv1 = DoubleConv(64, 64)
        self.add_conv2 = DoubleConv(64, 64)

        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        # if scale:
        #     self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


    def forward(self, x, mem_x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)

        x, skip1_out, skip2_out, skip3_out, skip4_out = self.resnet((x, mem_x))
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
        x = self.m1(x)
        x = self.add_conv1(x)
        x = self.m2(x)
        x = self.add_conv2(x)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class MemUNet(nn.Module):
    """UNet Architecture with membrane feature information"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, scale=False):
        """Initialize the UNet model"""
        super(MemUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv1_mem = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2_mem = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)
        self.attend = attend
        if scale:
            self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling
        if attend:
            self.attention1 = nn.MultiheadAttention(512*512, 4, dropout=0.2)
            self.attention2 = nn.MultiheadAttention(256*256, 4, dropout=0.2)


    def forward(self, x, mem_x):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width) 
        x_mem, skip1_out_mem = self.down_conv1_mem(mem_x)   
        x, skip2_out = self.down_conv2(x+x_mem) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        x_mem, skip2_out_mem = self.down_conv2_mem(x_mem)
        if self.three: x = x.squeeze(-3), x_mem.squeeze(-3)   
        x, skip3_out = self.down_conv3(x+x_mem) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        if self.three: 
            #attention_mode???
            skip1_out = torch.mean(skip1_out, dim=2)
            skip2_out = torch.mean(skip2_out, dim=2)
        x = self.up_conv2(x, skip2_out/2 + skip2_out_mem/2) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out/2 + skip1_out_mem/2) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class ExtendUNet(nn.Module):
    """UNet Architecture with membrane feature information"""
    def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, scale=False):
        """Initialize the UNet model"""
        super(ExtendUNet, self).__init__()
        self.three = three
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2 = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels

        self.down_conv1_pimg = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2_pimg = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels
        self.down_conv1_pmask = DownBlock(1, 64, three=three) # 3 input channels --> 64 output channels
        self.down_conv2_pmask = DownBlock(64, 128, three=three) # 64 input channels --> 128 output channels

        self.down_conv3 = DownBlock(128, 256) # 128 input channels --> 256 output channels
        self.down_conv4 = DownBlock(256, 512) # 256 input channels --> 512 output channels
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode) # 512 + 1024 input channels --> 512 output channels
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128+ 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x, pimg, pmask):
        """Forward pass of the UNet model
        x: (16, 1, 512, 512)
        """
        # print(x.shape)
        x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width) 
        x_pimg, skip1_out_pimg = self.down_conv1_pimg(pimg)   
        x_pmask, skip1_out_pmask = self.down_conv1_pmask(pmask)   


        x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
        x_pimg, skip2_out_pimg = self.down_conv2_pimg(x_pimg)   
        x_pmask, skip2_out_pmask = self.down_conv2_pmask(x_pmask)   

        x, skip3_out = self.down_conv3(x+x_pimg+x_pmask) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
        x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
        x = self.double_conv(x) # x: (16, 1024, 32, 32)
        x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
        x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
        x = self.up_conv2(x, skip2_out+skip2_out_pimg+skip2_out_pmask) # x: (16, 128, 256, 256)
        x = self.up_conv1(x, skip1_out+skip1_out_pimg+skip1_out_pmask) # x: (16, 64, 512, 512)
        x = self.conv_last(x) # x: (16, 1, 512, 512)
        return x

class PyramidPooling(nn.Module):

    def __init__(self, levels, channels=1, mode="max", method='spatial', td=False):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param channels defines the number of "color" channels in the data (used to determine dimensionality)
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :param method defines whether spatial or temporal pyramid pooling is used

        :returns a tensor vector with shape [batch x channels x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super().__init__()
        assert all(isinstance(_, int) for _ in levels)
        assert all(_ > 0 for _ in levels)
        self.levels = levels
        self.channels = channels
        self.mode = mode

        assert method.lower() in ['spatial', 'temporal']
        self.method = method.lower()

    def get_output_size(self):
        out = 0
        for level in self.levels:
            out += level * (level if self.method == 'spatial' else 1)
        return out
    
    def forward(self, x, compress_single_dimensions=True):
        assert isinstance(x, torch.Tensor)
        assert 2 <= len(x.shape) <= 4, "input x must be 2 dimensional (1 sample, 1 channel), 3 dimensional (1 sample, n channels | n samples, 1 channel) or 4 dimensional (n samples, n channels)"
        if len(x.shape) == 2:
            n_samples = 1
            assert self.channels == 1, "2 dimensional input passed when self.channels == %d (implicit single channel passed with 2D)" % self.channels
        elif len(x.shape) == 3:
            if self.channels == 1:
                n_samples = x.shape[0]
            else:
                assert self.channels == x.shape[0], "%d channels specified but 3D input of shape (%d, h, w) passed (implicit single sample passed with 3D)" % (self.channels, x.shape[0])
                n_samples = 1
        elif len(x.shape) == 4:
            self.channels = x.shape[1]
            assert x.shape[1] == self.channels, "second dimension of x input must represent the image channels but dimension == %d and self.channels = %d" % (x.shape[1], self.channels)
            n_samples = x.shape[0]
        
        n = n_samples
        c = self.channels
        h = x.shape[-2]
        w = x.shape[-1]
        
        result = self.pool(x.reshape(n, c, h, w), self.levels, self.mode, self.method, n=n, c=c, h=h, w=w)
        if compress_single_dimensions:
            if n == 1 and c == 1:
                return result.reshape(self.get_output_size())
            elif n == 1:
                return result.reshape(c, self.get_output_size())
            elif c == 1:
                return result.reshape(n, self.get_output_size())
        return result


    @staticmethod
    def pool(previous_conv, levels, mode, method, n, c, h, w):
        """
        Static Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :param method defines whether "spatial" or "temporal" pooling is used

        :returns a tensor vector with shape [batch x channels x n],
                                            where n: sum(level ** p) for each level in levels where p == 1 
                                            if temporal else 2 if spatial
        """
        for i, level in enumerate(levels):
            w_kernel = level
            w_pad1 = int(math.floor((w - w) / 2))
            w_pad2 = int(math.ceil((w - w) / 2))

            if method == 'spatial':
                h_kernel = level
                h_pad1 = int(math.floor((h - h) / 2))
                h_pad2 = int(math.ceil((h - h) / 2))

                # assert w_pad1 + w_pad2 == (w_kernel * w - w) and \
                #     h_pad1 + h_pad2 == (h_kernel * h - h)

                padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                    mode='constant', value=0)
            elif method == 'temporal':
                h_kernel = h

                assert w_pad1 + w_pad2 == (w_kernel * level - w)

                padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                    mode='constant', value=0)
            
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)

            if i == 0:
                # spp = x.view(num_sample, -1)
                spp = x.reshape(n, c, -1)
            else:
                # spp = torch.cat((spp, x.view(num_sample, -1)), 1)
                spp = torch.cat((spp, x.reshape(n, c, -1)), 2)

        return spp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, channels=1, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"

                :returns (forward) a tensor vector with shape [batch x channels x n],
                                                    where n: sum(level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(
            levels, channels=channels, mode=mode, method='spatial')