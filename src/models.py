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
        
