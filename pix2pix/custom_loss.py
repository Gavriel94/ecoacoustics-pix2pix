import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomL1Loss(nn.Module):
    def __init__(self, padding_value=1.0):
        super(CustomL1Loss, self).__init__()
        self.padding_value = padding_value

    def forward(self, input, target):
        if input.size() != target.size():
            target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

        mask = (input != self.padding_value).float()
        loss = nn.L1Loss(reduction='none')(input, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()
