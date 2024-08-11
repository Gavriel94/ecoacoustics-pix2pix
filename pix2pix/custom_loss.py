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

    def test_custom_l1_loss(self):
        input_tensor = torch.tensor([
            [[0.5, 1.0, 0.3],
             [1.0, 0.7, 1.0],
             [0.2, 1.0, 0.6]]
        ]).unsqueeze(0)
        target_tensor = torch.tensor([
            [[0.6, 1.0, 0.4],
             [1.0, 0.8, 1.0],
             [0.3, 1.0, 0.5]]
        ]).unsqueeze(0)

        loss = self.forward(input_tensor, target_tensor)

        non_padding_elements = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.6])
        target_elements = torch.tensor([0.6, 0.4, 0.8, 0.3, 0.5])
        expected_loss = F.l1_loss(non_padding_elements, target_elements, reduction='mean')

        print('loss', loss)
        print('expected loss', expected_loss)
        assert torch.isclose(loss, expected_loss, rtol=1e-4)