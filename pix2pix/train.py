import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

from . import config as cfg
from . import utilities


def remove_tensor_padding(image_tensor, original_size):
    _, padded_height, padded_width = image_tensor.shape
    original_height, original_width = original_size

    start_h = min(0, (padded_height - original_height) // 2)
    start_w = min(0, (padded_width - original_width) // 2)

    filtered = image_tensor[:, start_h:start_h+original_height, start_w:start_w+original_width]

    print(f'start_h = ({padded_height} - {original_height}) // 2', f'{(padded_height - original_height) // 2}')
    print(f'start_w = ({padded_width} - {original_width}) // 2', f'{(padded_width - original_width) // 2}')

    """
    start_h = (4096 - 2584) // 2 756
    start_w = (2048 - 2049) // 2 -1
    """

    print('REMOVING TENSOR PADDING')
    print('padded_height', padded_height)
    print('padded_width', padded_width)
    print('original_height', original_height)
    print('original_width', original_width)

    print('image_tensor.shape', image_tensor.shape)
    print('original_size', original_size)
    print('start_h', start_h)
    print('start_w', start_w)
    print('filtered', filtered)

    return filtered


def crop_padding(img_arr, original_dimensions):
    if img_arr.ndim == 3 and img_arr.shape[0] == 1:
        img_arr = img_arr.squeeze(0)
    original_height, original_width = original_dimensions
    mask = img_arr != 1.0
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    ymax = min(ymax + 1, original_height)
    xmax = min(xmax + 1, original_width)
    cropped = img_arr[ymin:ymax+1, xmin:xmax+1]
    return cropped


def custom_l1_loss(input, target, padding_value=1.0):
    if input.size() != target.size():
        target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

    mask = (input != padding_value).float()
    loss = nn.L1Loss(reduction='none')(input, target)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def train_model(discriminator, generator, data_loader, optim_discriminator, optim_generator, l1_loss, bce_logits):
    run_name = f"pix2pix/evaluation/run-{len(os.listdir('pix2pix/evaluation'))}"
    disc_loss, gen_loss, l1_loss = [], [], []
    for epoch in range(cfg.NUM_EPOCHS):
        discriminator.train()
        generator.train()
        train_loader_tqdm = tqdm(data_loader, leave=True)
        for idx, (input_img, target_img) in enumerate(train_loader_tqdm):
            save_path = f'{run_name}/epoch-{epoch}/batch_idx-{idx}/'
            os.makedirs(save_path, exist_ok=True)
            utilities.save_tensor(input_img, os.path.join(save_path, 'input.png'))
            utilities.save_tensor(target_img, os.path.join(save_path, 'target.png'))

            input_img, target_img = input_img.to(cfg.DEVICE), target_img.to(cfg.DEVICE)
            print(input_img)

            # train discriminator
            generated_image = generator(input_img)
            utilities.save_tensor(generated_image, os.path.join(save_path, 'generated.png'))
            D_real = discriminator(input_img, target_img)
            D_fake = discriminator(input_img, generated_image.detach())
            D_real_loss = bce_logits(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            optim_discriminator.zero_grad()
            D_loss.backward()
            optim_discriminator.step()

            # train generator
            D_fake = discriminator(input_img, generated_image)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))
            L1 = custom_l1_loss(generated_image, target_img) * cfg.L1_LAMBDA
            G_loss = G_fake_loss + L1

            optim_generator.zero_grad()
            G_loss.backward()
            optim_generator.step()

            train_loader_tqdm.set_postfix({
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'L1_loss': L1.item()
            })
            disc_loss.append(D_loss.item())
            gen_loss.append(G_loss.item())
            l1_loss.append(L1.item())

    return disc_loss, gen_loss, l1_loss
