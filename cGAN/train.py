import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from . import config as cfg


def custom_l1_loss(input, target, padding_value=0):
    if input.size() != target.size():
        target = F.interpolate(target, size=input.size()[2:], mode='bilinear', align_corners=False)

    mask = (input != padding_value).float()
    loss = nn.L1Loss(reduction='none')(input, target)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def train_model(disc, gen, train_loader, optim_disc, optim_gen, l1_loss, bce_logits):
    disc_loss, gen_loss, l1_loss = [], [], []
    for epoch in range(cfg.NUM_EPOCHS):
        disc.train()
        gen.train()
        train_loader_tqdm = tqdm(train_loader, leave=True)
        for idx, (input_img, target_img) in enumerate(train_loader_tqdm):
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

    return disc_loss, gen_loss, l1_loss
