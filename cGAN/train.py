import torch
from tqdm import tqdm
from . import config as cfg


def train_model(discriminator,
                generator,
                train_loader,
                optim_discriminator,
                optim_generator,
                l1_loss,
                bce_w_logits_loss):
    print('training')
    for epoch in range(cfg.NUM_EPOCHS):
        discriminator.train()
        generator.train()
        loop = tqdm(train_loader, leave=True)
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)

            # train discriminator
            with torch.autocast(device_type=cfg.DEVICE.type):
                y_fake = generator(x)
                D_real = discriminator(x, y)
                D_fake = discriminator(x, y_fake.detach())
                D_real_loss = bce_w_logits_loss(D_real, torch.ones_like(D_real))
                D_fake_loss = bce_w_logits_loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            optim_discriminator.zero_grad()
            D_loss.backward()
            optim_discriminator.step()

            # train generator
            with torch.autocast(device_type=cfg.DEVICE.type):
                D_fake = discriminator(x, y_fake)
                G_fake_loss = bce_w_logits_loss(D_fake, torch.ones_like(D_fake))
                L1 = l1_loss(y_fake, y) * cfg.L1_LAMBDA
                G_loss = G_fake_loss + L1

            optim_generator.zero_grad()
            G_loss.backward()
            optim_generator.step()
