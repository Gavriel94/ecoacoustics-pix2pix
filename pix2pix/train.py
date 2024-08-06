import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from . import config as cfg


def rescale_image(image, original_width, original_height):
    image = (image * 255).astype(np.uint8)
    image = image.reshape(image.shape[0], image.shape[1])
    return np.array(Image.fromarray(image).resize((original_width, original_height), Image.LANCZOS))


def compare_target_to_generated(epoch, real_images, generated_images, original_dimensions, initial_time, output_dir: str = None):
    if not output_dir:
        output_dir = os.path.join('pix2pix', 'evaluation', f'{datetime.now().strftime("%d-%m-%Y")}', 'generated_images')
    os.makedirs(output_dir, exist_ok=True)

    # Normalize the images to [0, 1] range for saving
    real_images = (real_images + 1) / 2.0
    generated_images = (generated_images + 1) / 2.0

    batch_size = real_images.size(0)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 5))

    for i, (real_image, generated_image) in enumerate(zip(real_images, generated_images)):
        real_img = real_image.permute(1, 2, 0).cpu().detach().numpy()
        gen_img = generated_image.permute(1, 2, 0).cpu().detach().numpy()

        original_height, original_width = original_dimensions

        real_img_rescaled = rescale_image(real_img, original_width, original_height)
        gen_img_rescaled = rescale_image(gen_img, original_width, original_height)

        if batch_size == 1:
            ax_real, ax_gen = axes
        else:
            ax_real, ax_gen = axes[i]
        ax_real.imshow(real_img)
        ax_real.set_title('Real Image')
        ax_real.axis('off')

        ax_gen.imshow(gen_img)
        ax_gen.set_title('Generated Image')
        ax_gen.axis('off')

        # save individual images
        real_img_path = os.path.join(output_dir, f'target_{epoch}_{i}.png')
        gen_img_path = os.path.join(output_dir, f'generated_{epoch}_{i}.png')
        real_img_pil = Image.fromarray(real_img_rescaled).save(real_img_path)
        gen_img_pil = Image.fromarray(gen_img_rescaled).save(gen_img_path)

    plt.tight_layout()
    os.makedirs(f"{output_dir}/compare_batch/", exist_ok=True)
    plt.savefig(f"{output_dir}/compare_batch/epoch_{epoch}.png")
    plt.close(fig)


def custom_l1_loss(input, target, padding_value=0):
    if input.size() != target.size():
        target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

    mask = (input != padding_value).float()
    loss = nn.L1Loss(reduction='none')(input, target)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def train_model(disc, gen, train_loader, optim_disc, optim_gen, l1_loss, bce_logits):
    initial_time = datetime.now().strftime("%H-%M-%S")
    disc_loss, gen_loss, l1_loss = [], [], []
    for epoch in range(cfg.NUM_EPOCHS):
        disc.train()
        gen.train()
        train_loader_tqdm = tqdm(train_loader, leave=True)
        for idx, (input_img, target_img, original_dimensions) in enumerate(train_loader_tqdm):
            input_img, target_img = input_img.to(cfg.DEVICE), target_img.to(cfg.DEVICE)

            # train discriminator
            y_fake = gen(input_img)
            D_real = disc(input_img, target_img)
            D_fake = disc(input_img, y_fake.detach())
            D_real_loss = bce_logits(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            optim_disc.zero_grad()
            D_loss.backward()
            optim_disc.step()

            # train generator
            D_fake = disc(input_img, y_fake)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))
            L1 = custom_l1_loss(y_fake, target_img) * cfg.L1_LAMBDA
            G_loss = G_fake_loss + L1

            optim_gen.zero_grad()
            G_loss.backward()
            optim_gen.step()

            train_loader_tqdm.set_postfix({
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'L1_loss': L1.item()
            })
            disc_loss.append(D_loss.item())
            gen_loss.append(G_loss.item())
            l1_loss.append(L1.item())

        original_height = int(original_dimensions[0].item())
        original_width = int(original_dimensions[1].item())
        original_dims = (original_height, original_width)
        compare_target_to_generated(epoch,
                                    real_images=target_img,
                                    generated_images=y_fake,
                                    original_dimensions=original_dims,
                                    initial_time=initial_time)

    return disc_loss, gen_loss, l1_loss
