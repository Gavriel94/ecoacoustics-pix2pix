import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

from . import utilities as ut


def custom_l1_loss(input, target, padding_value=1.0):
    if input.size() != target.size():
        target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

    mask = (input != padding_value).float()
    loss = nn.L1Loss(reduction='none')(input, target)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def train(discriminator, generator, data_loader, optim_discriminator, optim_generator, l1_loss, l1_lambda, bce_logits, num_epochs, device, save_dir: str, accumulation_steps: int = 1, display_epoch: int = 5):
    os.makedirs(f'{save_dir}/evaluation', exist_ok=True)
    run_name = f"{save_dir}/evaluation/run_{len(os.listdir(f'{save_dir}/evaluation')) + 1}"
    os.makedirs(run_name, exist_ok=True)
    disc_loss, gen_loss, l1_loss = [], [], []
    for epoch in range(num_epochs):
        discriminator.train()
        generator.train()
        data_loader_tqdm = tqdm(data_loader, leave=True)
        for idx, (input_img, target_img, original_size, padding_coords, img_name) in enumerate(data_loader_tqdm):
            input_img, target_img = input_img.to(device), target_img.to(device)

            # train discriminator
            generated_img = generator(input_img)  # synthesise image
            D_real = discriminator(input_img, target_img)
            D_fake = discriminator(input_img, generated_img.detach())
            # calculate losses
            D_real_loss = bce_logits(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
            # calculate average of losses
            D_loss = (D_real_loss + D_fake_loss) / 2
            # normalise loss based on accumulation steps
            D_loss_norm = D_loss / accumulation_steps
            # backpropagate
            D_loss_norm.backward()

            # train generator
            D_fake = discriminator(input_img, generated_img)  # disc output for synth pair (input, generated)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))  # BCE for synth images (gen tries to fool dis, should be close to 1)
            L1 = custom_l1_loss(generated_img, target_img) * l1_lambda # L1 loss between synth and target images encourage visual similarity
            G_loss = G_fake_loss + L1  # generator loss is a combination of adverserial (G_fake_loss) and L1 loss
            # backpropagate and update generator weights
            G_loss_norm = G_loss / accumulation_steps
            G_loss_norm.backward()

            data_loader_tqdm.set_postfix({
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'L1_loss': L1.item()
            })
            disc_loss.append(D_loss.item())
            gen_loss.append(G_loss.item())
            l1_loss.append(L1.item())

            if (idx + 1) % accumulation_steps == 0:
                optim_discriminator.step()
                optim_generator.step()
                optim_discriminator.zero_grad()
                optim_generator.zero_grad()

            if idx % display_epoch == 0:
                save_path = f'{run_name}/'
                # os.makedirs(save_path, exist_ok=True)
                img_name = str(img_name[idx])  # unpack from tuple

                # crop and save input image
                input_cropped = ut.remove_padding(input_img, original_size,
                                                  padding_coords, is_target=False)
                input_path = os.path.join(run_name, f'e{epoch}_b{idx}_i_{img_name}')
                ut.save_tensor_as_img(input_cropped, input_path)

                # crop and save target image
                target_cropped = ut.remove_padding(target_img, original_size,
                                                  padding_coords, is_target=True)
                target_path = os.path.join(run_name, f'e{epoch}_b{idx}_t_{img_name}')
                ut.save_tensor_as_img(target_cropped, target_path)

                generated_cropped = ut.remove_padding(generated_img, original_size,
                                                      padding_coords, is_target=False)
                generated_path = os.path.join(run_name, f'e{epoch}_b{idx}_g_{img_name}')
                ut.save_tensor_as_img(generated_cropped, generated_path)

        # final update if number of batches is not divisible by accumulation_steps
        if len(data_loader) % accumulation_steps != 0:
            optim_discriminator.step()
            optim_generator.step()
            optim_discriminator.zero_grad()
            optim_generator.zero_grad()

    ut.plot_loss(disc_loss, gen_loss, l1_loss, run_name)
