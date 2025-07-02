"""
Complete Models and Components for Gap Junction Segmentation.
Tommy Tang
June 1, 2025
"""

#Libraries
import torch 
import torch.nn as nn
import torch.nn.functional as F

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
        
#Models

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
        
