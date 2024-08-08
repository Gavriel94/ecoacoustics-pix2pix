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


def train_model(discriminator, generator, data_loader, optim_discriminator, optim_generator, l1_loss, l1_lambda, bce_logits, num_epochs, device):
    os.makedirs('data/evaluation', exist_ok=True)
    run_name = f"data/evaluation/run-{len(os.listdir('data/evaluation'))}"
    disc_loss, gen_loss, l1_loss = [], [], []
    for epoch in range(num_epochs):
        discriminator.train()
        generator.train()
        train_loader_tqdm = tqdm(data_loader, leave=True)
        for idx, (input_img, target_img, original_size, padding_coords, img_names) in enumerate(train_loader_tqdm):
            save_path = f'{run_name}/epoch_{epoch}/batch_idx_{idx}/'
            os.makedirs(save_path, exist_ok=True)

            input_img, target_img = input_img.to(device), target_img.to(device)

            # train discriminator
            generated_image = generator(input_img)  # synthesise image
            D_real = discriminator(input_img, target_img)  # dis output for real pair (input, target)
            D_fake = discriminator(input_img, generated_image.detach())  # dis output for synth pair (input, generated)
            # calculate losses
            D_real_loss = bce_logits(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
            # calculate average of losses
            D_loss = (D_real_loss + D_fake_loss) / 2
            # backpropagate and update disciminator weights
            optim_discriminator.zero_grad()
            D_loss.backward()
            optim_discriminator.step()

            # train generator
            D_fake = discriminator(input_img, generated_image)  # disc output for synth pair (input, generated)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))  # BCE for synth images (gen tries to fool dis, should be close to 1)
            L1 = custom_l1_loss(generated_image, target_img) * l1_lambda # L1 loss between synth and target images encourage visual similarity
            G_loss = G_fake_loss + L1  # generator loss is a combination of adverserial (G_fake_loss) and L1 loss
            # backpropagate and update generator weights
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
            
            if idx % 5 == 0:
                ut.save_tensor(
                    ut.remove_padding(input_img, original_size,
                                      padding_coords, is_target=False),
                    os.path.join(save_path, f"{img_names[idx].replace('.png', '_input.png')}")
                )
                ut.save_tensor(
                    ut.remove_padding(target_img, original_size,
                                      padding_coords, is_target=True),
                    os.path.join(save_path, f"{img_names[idx].replace('.png', '_target.png')}")
                )
                ut.save_tensor(
                    ut.remove_padding(generated_image, original_size,
                                      padding_coords, is_target=False),
                    os.path.join(save_path, f"{img_names[idx].replace('.png', '_generated.png')}")
                )

    return disc_loss, gen_loss, l1_loss
