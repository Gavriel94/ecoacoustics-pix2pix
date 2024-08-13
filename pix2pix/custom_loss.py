"""
Custom L1 loss that ignores a padding value while computing a loss value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pix2PixLoss(nn.Module):
    """
    Custom L1 loss that ignores padding and incorporates intensity awareness.

    Calculates a combination of L1 loss and intensity-based loss between
    input and target tensors, ignoring areas where the input matches the padding value.
    It handles size mismatches by interpolating the target to match the input size.

    Args:
        padding_value (int): Value to ignore when computing loss.
        alpha (float, optional): Intensity awareness weight when combining loss. Default is 0.5.
    """
    def __init__(self, padding_value=1.0, alpha=0.5):
        super(Pix2PixLoss, self).__init__()
        self.padding_value = padding_value  # value to ignore
        self.alpha = alpha  # weight for intensity loss

    def forward(self, input, target):
        """
        Compute the combined L1 and intensity-aware loss.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if input.size() != target.size():
            # resize target tensor to match input
            target = F.interpolate(target, size=input.size()[2:],
                                   mode='bilinear', align_corners=False)

        mask = (input != self.padding_value).float()

        # compute L1 loss
        l1_loss = nn.L1Loss(reduction='none')(input, target)
        masked_l1_loss = l1_loss * mask
        l1_loss_value = masked_l1_loss.sum() / mask.sum()

        # compute intensity loss
        input_intensity = (torch.mean(input * mask, dim=[1, 2, 3])
                           / torch.mean(mask, dim=[1, 2, 3]))

        target_intensity = (torch.mean(target * mask, dim=[1, 2, 3])
                            / torch.mean(mask, dim=[1, 2, 3]))

        intensity_loss = F.mse_loss(input_intensity, target_intensity)

        # combine losses
        total_loss = (1 - self.alpha) * l1_loss_value + self.alpha * intensity_loss
        return total_loss

    def test_custom_l1_loss(self):
        """
        Compare a padded and unpadded tensor to see if the loss values
        equate.
        """
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

        print(f'Expected loss: {expected_loss}')
        print(f'Actual loss  : {loss}')
        assert torch.isclose(loss, expected_loss, rtol=1e-4)
