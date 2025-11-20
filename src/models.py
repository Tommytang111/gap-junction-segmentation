"""
Models and Related Components Necessary for Gap Junction Segmentation.
Tommy Tang
June 1, 2025
"""

#Libraries
import os
import cv2
import numpy as np
from typing import Union
from PIL import Image
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from utils import filter_pixels, resize_image
from entities.entity_detection_2d import GapJunctionEntityDetector2D

#DATASETS
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, augmentation=None, data_size=(512, 512), train=True):
        self.image_paths = sorted([os.path.join(images, img) for img in os.listdir(images)])
        self.label_paths = sorted([os.path.join(labels, lbl) for lbl in os.listdir(labels)])
        self.augmentation = augmentation
        self.data_size = data_size
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #Read image, label
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        #Apply resizing with padding if image is not expected size and then convert back to ndarray
        if (image.shape[0] != self.data_size[0]) or (image.shape[1] != self.data_size[1]): 
            image = np.array(resize_image(image, self.data_size[0], self.data_size[1], pad_clr=(0,0,0), channels=False))
            label = np.array(resize_image(label, self.data_size[0], self.data_size[1], pad_clr=(0,0,0), channels=False))

        #Convert label to binary for model classification
        label[label > 0] = 1
            
        #Filter small out small groups of pixels (annotation mistakes)
        label = filter_pixels(label, size_threshold=10)
        
        #Apply augmentation if provided
        if self.augmentation and self.train:
                augmented = self.augmentation(image=image, mask=label)
                image = augmented['image']
                label = augmented['mask']

        #Add entity recognition clause later if needed
        
        # Convert to tensors if not already converted from augmentation
        if not torch.is_tensor(image):
            image = ToTensor()(image).float()
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long()

        return image, label
    
class TrainingDataset3D(torch.utils.data.Dataset):
    def __init__(self, volumes, labels, augmentation=None, data_size=(9, 512, 512), train=True):
        self.volume_paths = sorted([os.path.join(volumes, vol) for vol in os.listdir(volumes)])
        self.label_paths = sorted([os.path.join(labels, lbl) for lbl in os.listdir(labels)])
        self.augmentation = augmentation
        self.data_size = data_size
        self.train = train

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx):
        #Read volume, label
        volume = np.load(self.volume_paths[idx])
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)

        #Convert label to binary for model classification
        label[label > 0] = 1
            
        #Filter small out small groups of pixels (annotation mistakes)
        label = filter_pixels(label, size_threshold=10)
        
        #Apply augmentation if provided
        if self.augmentation:
            #Make additional targets dict
            additional_targets = {}
            for i in range(1, volume.shape[0]):
                target_key = f'image{i}'
                additional_targets[target_key] = 'image'
                
            #Update albumentations pipeline with additional targets for all slices in volume
            self.augmentation.add_targets(additional_targets)

            #Prepare data dictionary with all slices, adding an extra channel dimension at the end
            #Note: albumentations Compose is supposed to add a channel dimension automatically and then remove it after 
            #augmentation, but it keeps crashing the script so I do it manually here.
            aug_data = {'image': volume[0][..., None], 'mask': label[..., None]}  # First slice as main image
            for i in range(1, volume.shape[0]):
                target_key = f'image{i}'
                aug_data[target_key] = volume[i][..., None]

            #Apply augmentation once to all slices
            augmented = self.augmentation(**aug_data)

            #Reconstruct volume from augmented slices
            #Note: When mask is provided, extra channel dimension is at end (dim = -1), when it's not provided, it is first (dim=0).
            augmented_slices = [np.squeeze(augmented['image'], -1)]  # First slice, remove channel dimension
            for i in range(1, volume.shape[0]):
                augmented_slices.append(np.squeeze(augmented[f'image{i}'], -1))

            volume = np.stack(augmented_slices, axis=0)
            label = np.squeeze(augmented['mask'], -1)

        # Convert to tensors if not already converted from augmentation
        if not torch.is_tensor(volume):
            # Ensure volume shape is (channels=1, depth, height, width)
            if volume.ndim == 3:
                volume = volume[None, ...]  # Add channel dimension: (1, D, H, W)
            elif volume.ndim == 4 and volume.shape[0] == 1:
                pass # Already in correct format
            else:
                raise ValueError(f"Unexpected volume shape: {volume.shape}")
            volume = torch.from_numpy(volume.astype(np.float32))
        else:
            volume = volume.float()
        #Label
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long()
            
        return volume, label

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images, augmentation=None, data_size=(512, 512)):      
        self.image_paths = sorted([os.path.join(images, img) for img in os.listdir(images)])
        self.augmentation = augmentation
        self.data_size = data_size

    def __len__(self):
        # return length of 
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #Read image, label
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {self.image_paths[idx]}")

        #Apply resizing with padding if image is not expected size and then convert back to ndarray
        if (image.shape[0] != self.data_size[0]) or (image.shape[1] != self.data_size[1]): 
            image = np.array(resize_image(image, self.data_size[0], self.data_size[1], pad_clr=(0,0,0), channels=False))
        
        #Apply augmentation if provided
        if self.augmentation:
                augmented = self.augmentation(image=image)
                image = augmented['image']
        
        # Convert to tensors if not already converted from augmentation
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.astype(np.float32))

        #Add batch dimension to image
        return image

class TestDataset3D(torch.utils.data.Dataset):
    def __init__(self, volumes, augmentation=None, data_size=(9, 512, 512)):
        self.volume_paths = sorted([os.path.join(volumes, vol) for vol in os.listdir(volumes)])
        self.augmentation = augmentation
        self.data_size = data_size
        
    def __len__(self):
        # return length of 
        return len(self.volume_paths)
    
    def __getitem__(self, idx):
        #Read volume, label
        volume = np.load(self.volume_paths[idx])
        if volume is None:
            raise ValueError(f"Failed to read volume: {self.volume_paths[idx]}")

        #Apply augmentation if provided
        if self.augmentation:
            #Make additional targets dict
            additional_targets = {}
            for i in range(1, volume.shape[0]):
                target_key = f'image{i}'
                additional_targets[target_key] = 'image'
                
            #Update albumentations pipeline with additional targets for all slices in volume
            self.augmentation.add_targets(additional_targets)

            #Prepare data dictionary with all slices, adding an extra channel dimension at the end
            #Note: albumentations Compose is supposed to add a channel dimension automatically and then remove it after 
            #augmentation, but the removal doesn't work so I do it manually here.
            aug_data = {'image': volume[0][..., None]}  # First slice as main image
            for i in range(1, volume.shape[0]):
                target_key = f'image{i}'
                aug_data[target_key] = volume[i][..., None]

            #Apply augmentation once to all slices
            augmented = self.augmentation(**aug_data)

            #Reconstruct volume from augmented slices
            #Note: When mask is provided, extra channel dimension is at end (dim = -1), when it's not provided, it is first (dim=0).
            augmented_slices = [np.squeeze(augmented['image'], 0)]  # First slice, remove channel dimension
            for i in range(1, volume.shape[0]):
                augmented_slices.append(np.squeeze(augmented[f'image{i}'], 0))

            volume = np.stack(augmented_slices, axis=0)

        # Convert to tensors if not already converted from augmentation
        if not torch.is_tensor(volume):
            # Ensure volume shape is (channels=1, depth, height, width)
            if volume.ndim == 3:
                volume = volume[None, ...]  # Add channel dimension: (1, D, H, W)
            elif volume.ndim == 4 and volume.shape[0] == 1:
                pass # Already in correct format
            else:
                raise ValueError(f"Unexpected volume shape: {volume.shape}")
            volume = torch.from_numpy(volume.astype(np.float32))
        else:
            volume = volume.float()
            
        return volume
        
#Models and Building Blocks
class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None, three=False, dropout=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False) if not three else nn.Conv3d(in_channels, mid_channels, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm2d(mid_channels) if not three else nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False) if not three else nn.Conv3d(mid_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):
    """Triple convolution block used in UNet."""

    def __init__(self, in_channels, out_channels, mid_channels=None, three=False, dropout=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False) if not three else nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels) if not three else nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False) if not three else nn.Conv3d(mid_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False) if not three else nn.Conv3d(mid_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.triple_conv(x)

class DownBlock(nn.Module):
    """Double convolution followed by max pooling"""

    def __init__(self, in_channels, out_channels, three=False, dropout=0):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, three=three, dropout=dropout)
        self.down_sample = nn.MaxPool2d(2) if not three else nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)

class UpBlock(nn.Module):
    """Upsampling followed by double convolution"""

    def __init__(self, in_channels, out_channels, up_sample_mode, three=False, dropout=0):
        super().__init__()

        if up_sample_mode =='conv_transpose':
            if three: 
                self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            )
            else: 
                self.up_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        elif up_sample_mode == 'bilinear':
            if three:
                self.up_sample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
            else:
                self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported up_sample_mode: {up_sample_mode}, choose from 'conv_transpose' or 'bilinear'.")
        
        self.double_conv=DoubleConv(in_channels, out_channels, three=three, dropout=dropout)

    def forward(self, up_input, skip_input):
        """
        Concatenate the upsampled input with skip_input along the channel dimension and apply double convolution. Since padding
        is set to 1 in the convolution layers, up_input and skip_input should be the same size before concatenation.
        """
        x = self.up_sample(up_input)
        x = torch.cat([skip_input, x], dim=1)
        return self.double_conv(x)

class OutConv(nn.Module):
    """
    Output convolution layer. If 3D-2D Unet, uses a 3D convolution with kernel size (classes, 1, 1), where classes is 
    depth, to have an output that retains image dimensions with as many channels as classes.
    """

    def __init__(self, in_channels, classes=2, three=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, classes, kernel_size=1) if not three else nn.Conv3d(in_channels, classes, kernel_size=(9,1,1))

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet model for image segmentation
    
    Args:
        n_channels (int): Number of input channels
        classes (int): Number of output classes
        up_sample_mode (str): Upsampling mode, either 'conv_transpose' or 'bilinear'
        three (bool): If True, uses 3D convolutions; otherwise, uses 2D convolutions
        dropout (float): Dropout rate for regularization
    """

    def __init__(self, n_channels=1, classes=2, up_sample_mode='conv_transpose', three=False, dropout=0):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.down1 = DownBlock(n_channels, 64, three=three, dropout=dropout)
        self.down2 = DownBlock(64, 128, three=three, dropout=dropout)
        self.down3 = DownBlock(128, 256, three=three, dropout=dropout)
        self.down4 = DownBlock(256, 512, three=three, dropout=dropout)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, three=three, dropout=dropout)
        
        # Decoder (Expansive Path)
        self.up1 = UpBlock((512 if up_sample_mode == 'conv_transpose' else 1024) + 512, 512, up_sample_mode, three=three, dropout=dropout)
        self.up2 = UpBlock((256 if up_sample_mode == 'conv_transpose' else 512) + 256, 256, up_sample_mode, three=three, dropout=dropout)
        self.up3 = UpBlock((128 if up_sample_mode == 'conv_transpose' else 256) + 128, 128, up_sample_mode, three=three, dropout=dropout)
        self.up4 = UpBlock((64 if up_sample_mode == 'conv_transpose' else 128) + 64, 64, up_sample_mode, three=three, dropout=dropout)

        # Output Layer
        self.output = OutConv(64, classes=classes, three=three)

    def forward(self, x):
        """
        Forward pass through the UNet model.
        x has dimensions (batch_size, n_channels, height, width) for 2D or (batch_size, n_channels, depth, height, width) for 3D.
        """
        # Encoder
        x1, skip_x = self.down1(x)
        #print("x1:", x1.shape, "skip_x:", skip_x.shape)
        x2, skip_x1 = self.down2(x1)
        #print("x2:", x2.shape, "skip_x1:", skip_x1.shape)
        x3, skip_x2 = self.down3(x2)
        #print("x3:", x3.shape, "skip_x2:", skip_x2.shape)
        x4, skip_x3 = self.down4(x3)
        #print("x4:", x4.shape, "skip_x3:", skip_x3.shape)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        #print("x5 (bottleneck):", x5.shape)
        
        # Decoder with skip connections
        x6 = self.up1(x5, skip_x3)
        #print("x6:", x6.shape)
        x7 = self.up2(x6, skip_x2)
        #print("x7:", x7.shape)
        x8 = self.up3(x7, skip_x1)
        #print("x8:", x8.shape)
        x9 = self.up4(x8, skip_x)
        #print("x9:", x9.shape)
        logits = self.output(x9)
        #print("logits:", logits.shape)
        return logits

#Previous Unet model definition for reference
# class DoubleConv(nn.Module):
#     """(Conv2d -> BN -> ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, three=False, dropout=0):
#         super(DoubleConv, self).__init__()
        
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if not three else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels) if not three else nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=False),
#             nn.Dropout(p=dropout)
#         )
  
#     def forward(self, x_in):
#         x = self.double_conv(x_in)
#         con_shape = x.shape
#         return x
    
# class DownBlock(nn.Module):
#     """Double Convolution followed by Max Pooling"""
#     def __init__(self, in_channels, out_channels, three=False, dropout=0):
#         super(DownBlock, self).__init__()
#         self.double_conv = DoubleConv(in_channels, out_channels, three=three, dropout=dropout)
#         self.down_sample = nn.MaxPool2d(2, stride=2) if not three else nn.MaxPool3d(2, stride=2)

#     def forward(self, x):
#         skip_out = self.double_conv(x)
#         down_out = self.down_sample(skip_out)
#         return (down_out, skip_out)

# class UpBlock(nn.Module):
#     """Up Convolution (Upsampling followed by Double Convolution)"""
#     def __init__(self, in_channels, out_channels, up_sample_mode, kernel_size=2, three=False, dropout=0):
#         super(UpBlock, self).__init__()
#         if up_sample_mode == 'conv_transpose':
#             if three: self.up_sample = nn.Sequential(
#                 nn.ConvTranspose3d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
#                 nn.BatchNorm3d(in_channels-out_channels),
#                 nn.ReLU())       
#             else: self.up_sample = nn.Sequential(
#                 nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=kernel_size, stride=2),
#                 nn.BatchNorm2d(in_channels-out_channels),
#                 nn.ReLU())
#         elif up_sample_mode == 'bilinear':
#             self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True, three=three)
#         else:
#             raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
#         self.double_conv = DoubleConv(in_channels, out_channels, three=three)

#     def forward(self, down_input, skip_input):
#         x = self.up_sample(down_input)
#         x = torch.cat([x, skip_input], dim=1)
#         return self.double_conv(x)
    
# class UNet(nn.Module):
#     """UNet Architecture"""
#     def __init__(self, out_classes=2, up_sample_mode='conv_transpose', three=False, attend=False, scale=False, dropout=0, classes=2):
#         """Initialize the UNet model"""
#         super(UNet, self).__init__()
#         self.three = three
#         self.up_sample_mode = up_sample_mode
#         self.dropout=dropout

#         # Downsampling Path
#         self.down_conv1 = DownBlock(1, 64, three=three) # 1 input channels --> 64 output channels
#         self.down_conv2 = DownBlock(64, 128, three=three, dropout=self.dropout) # 64 input channels --> 128 output channels
#         self.down_conv3 = DownBlock(128, 256, three=three, dropout=self.dropout) # 128 input channels --> 256 output channels
#         self.down_conv4 = DownBlock(256, 512, three=three, dropout=self.dropout) # 256 input channels --> 512 output channels
#         # Bottleneck
#         self.double_conv = DoubleConv(512, 1024, three=three, dropout=self.dropout)
#         # Upsampling Path
#         self.up_conv4 = UpBlock(512 + 1024, 512, three=three, up_sample_mode=self.up_sample_mode, dropout=self.dropout) # 512 + 1024 input channels --> 512 output channels
#         self.up_conv3 = UpBlock(256 + 512, 256, three=three, up_sample_mode=self.up_sample_mode, dropout=self.dropout)
#         self.up_conv2 = UpBlock(128 + 256, 128, three=three, up_sample_mode=self.up_sample_mode, dropout=self.dropout)
#         self.up_conv1 = UpBlock(64 + 128, 64, three=three, up_sample_mode=self.up_sample_mode, dropout=self.dropout)
#         # Final Convolution
#         self.conv_last = nn.Conv2d(64, 1 if classes == 2 else classes, kernel_size=1)
#         self.attend = attend
#         if scale:
#             self.s1, self.s2 = torch.nn.Parameter(torch.ones(1), requires_grad=True), torch.nn.Parameter(torch.ones(1), requires_grad=True) # learn scaling


#     def forward(self, x):
#         """Forward pass of the UNet model
#         x: (16, 1, 512, 512)
#         """
#         # print(x.shape)
#         x, skip1_out = self.down_conv1(x) # x: (16, 64, 256, 256), skip1_out: (16, 64, 512, 512) (batch_size, channels, height, width)    
#         x, skip2_out = self.down_conv2(x) # x: (16, 128, 128, 128), skip2_out: (16, 128, 256, 256)
#         if self.three: x = x.squeeze(-3)   
#         x, skip3_out = self.down_conv3(x) # x: (16, 256, 64, 64), skip3_out: (16, 256, 128, 128)
#         x, skip4_out = self.down_conv4(x) # x: (16, 512, 32, 32), skip4_out: (16, 512, 64, 64)
#         x = self.double_conv(x) # x: (16, 1024, 32, 32)
#         x = self.up_conv4(x, skip4_out) # x: (16, 512, 64, 64)
#         x = self.up_conv3(x, skip3_out) # x: (16, 256, 128, 128)
#         if self.three: 
#             skip1_out = torch.mean(skip1_out, dim=2)
#             skip2_out = torch.mean(skip2_out, dim=2)
#         x = self.up_conv2(x, skip2_out) # x: (16, 128, 256, 256)
#         x = self.up_conv1(x, skip1_out) # x: (16, 64, 512, 512)
#         x = self.conv_last(x) # x: (16, 1, 512, 512)
#         return x
        
#LOSS FUNCTIONS
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

class GenDLoss(nn.Module):
    def __init__(self):
        super(GenDLoss, self).__init__()
    
    def forward(self, inputs, targets, loss_fn=None, fn_reweight=None):
        inputs = nn.Sigmoid()(inputs)
        
        # Handle 2-channel outputs (select the foreground channel)
        if inputs.shape[1] == 2:
            inputs = inputs[:, 1:2, :, :]  # Take only the foreground probability
        
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)

        inputs = torch.stack([inputs, 1-inputs], dim=-1)
        targets = torch.stack([targets, 1-targets], dim=-1)

        weights = 1 / (torch.sum(torch.permute(targets, (0, 2, 1)), dim=-1).pow(2)+1e-6)
        targets, inputs = torch.permute(targets, (0, 2, 1)), torch.permute(inputs, (0, 2, 1))

        return torch.nanmean(1 - 2 * torch.nansum(weights * torch.nansum(targets * inputs, dim=-1), dim=-1)/\
                          torch.nansum(weights * torch.nansum(targets + inputs, dim=-1), dim=-1))


class GenDLossEntity(nn.Module): 
    def __init__(self):
        super(GenDLossEntity, self).__init__()
    
    def forward(self, preds, targets, loss_mask=None, mito_mask=None, classes=2, **kwargs):
        """
        Compute Generalized Dice Loss for entities.
        :param preds: Predicted logits from the model 
        :param targets: Ground truth labels, shape 
        
        :return: Generalized Dice Loss value.
        """
        entity_detector = GapJunctionEntityDetector2D()
        pred_entities = entity_detector.extract_entities_2d(preds)
        gt_entities = entity_detector.extract_entities_2d(targets)
        
        ## idea is to find matching entities, then separate predicted image and gt image into 2 channels:
        ## - foreground channel = all pixels that belong to matched entities
        ## - background channel = all other pixels (including unmatched entities -- so if the model detects an entity but
        ##  it doesn't match to any gt entity, it's considered background)

class GeneralizedIoULoss(nn.Module):
    """
        Compute Generalized IoU Loss for entities.
        :param preds: Predicted logits from the model (tensor)
        :param targets: Ground truth labels (tensor)
        
        :return: Generalized IoU Loss value.
    """
    def __init__(self):
        super(GeneralizedIoULoss, self).__init__()
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], classes=2, **kwargs):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)

        intersection = torch.sum(targets * inputs, dim=-1)
        union = torch.sum(targets + inputs, dim=-1) - intersection
        iou = intersection / (union + 1e-6)

        # Compute enclosing box
        enclosing_min = torch.min(targets, inputs)
        enclosing_max = torch.max(targets, inputs)
        enclosing_area = torch.sum(enclosing_max - enclosing_min, dim=-1)

        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)

        return torch.mean(1 - giou)

class DistanceIoULoss(nn.Module):
    """
        Compute Distance IoU Loss for entities.
        :param preds: Predicted logits from the model 
        :param targets: Ground truth labels 
        
        :return: Distance IoU Loss value.
    """    
    def __init__(self):
        super(DistanceIoULoss, self).__init__()
    
    def forward(self, inputs, targets, loss_mask=[], mito_mask=[], classes=2, **kwargs):
        inputs = nn.Sigmoid()(inputs)
        targets, inputs = targets.view(targets.shape[0], -1), inputs.view(inputs.shape[0], -1)

        intersection = torch.sum(targets * inputs, dim=-1)
        union = torch.sum(targets + inputs, dim=-1) - intersection
        iou = intersection / (union + 1e-6)

        # Compute center points
        targets_center = torch.mean(targets, dim=-1)
        inputs_center = torch.mean(inputs, dim=-1)
        center_distance = torch.sum((targets_center - inputs_center) ** 2, dim=-1)

        # Compute enclosing box diagonal length
        enclosing_min = torch.min(targets, inputs)
        enclosing_max = torch.max(targets, inputs)
        enclosing_diagonal = torch.sum((enclosing_max - enclosing_min) ** 2, dim=-1)

        diou = iou - center_distance / (enclosing_diagonal + 1e-6)

        return torch.mean(1 - diou)

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

        
# Example usage and testing
if __name__ == "__main__":
    # Create model instance
    model = UNet(three=False, n_channels=1, classes=2, up_sample_mode='conv_transpose')  # 1 input channel (grayscale), 2 classes (background, gap junction)

    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy input
    x = torch.randn(1, 1, 512, 512)  # Batch size 1, 1 channel, 512x512 image
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
    # Example training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("\nModel created successfully!")
    print("Key features:")
    print("- Encoder-decoder architecture with skip connections")
    print("- Batch normalization for stable training")
    print("- ReLU activations")
    print("- Configurable input/output channels")
    print("- Option for bilinear upsampling or transposed convolutions")