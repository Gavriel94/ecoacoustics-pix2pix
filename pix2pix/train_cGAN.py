"""
Train the spectrogram translation conditional Generative Adversial Network.
"""

import os
import json

import torch
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from tqdm import tqdm

from . import utilities as ut


def train_cGAN(discriminator, generator,
               train_loader, validation_loader,
               optim_discriminator, optim_generator,
               custom_loss, loss_lambda,
               bce_logits, num_epochs,
               device, save_dir: str,
               accumulation_steps):
    """
    Train the conditional Generative Adverserial Network (cGAN).

    Evaluates the model at the end of each batch by running the cGAN on
    unseen data. Data used in the evaluation does not contribute to weight
    updates in the model.

    Pairs of generated and target images are saved during training so
    the quality of the output can be inspected in real time.

    An average of both peak signal to noise ratio (PSNR) and
    structural similarity index measure (SSIM) are computed to quantify
    how similar the synthesised image is to the target image.

    Args:
        discriminator (Discriminator): Discriminator model.
        generator (Generator): Generator model.
        train_loader (DataLoader): Training data.
        validation_loader (DataLoader): Validation data.
        optim_discriminator (optim.Adam): Discriminator optimiser.
        optim_generator (optim.Adam): Discriminator generator.
        custom_loss (Pix2PixLoss): Generator loss function.
        loss_lambda (int): Balance combined L1/Intensity with adverserial loss.
        bce_logits (nn.BCEWithLogitsLoss): Discriminator loss function.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device for tensor operations.
        save_dir (str): Output for evaluation data.
        accumulation_steps (int): Steps before updating gradients. Simulates batching.
    """
    runs_dir = os.path.join(save_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    run_num = int(len(os.listdir(runs_dir))) + 1
    run_name = os.path.join(runs_dir, f'run_{run_num}')
    os.makedirs(run_name, exist_ok=True)

    metrics = {
        'disc_losses': [],
        'gen_losses': [],
        'custom_loss_losses': [],
        'avg_psnrs': [],
        'avg_ssims': []
    }

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    print(f'Run {run_num}')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        discriminator.train()
        generator.train()
        data_loader_tqdm = tqdm(train_loader, leave=True)  # view progress bar while training
        for idx, (input_img, target_img,
                  original_size, padding_coords, img_names) in enumerate(data_loader_tqdm):
            input_img, target_img = input_img.to(device), target_img.to(device)

            # * train discriminator
            generated_img = generator(input_img)
            D_real, real_features = discriminator(input_img, target_img)
            D_fake, fake_features = discriminator(input_img, generated_img.detach())
            # calculate losses
            D_real_loss = bce_logits(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
            # calculate average of losses
            D_loss = (D_real_loss + D_fake_loss) / 2
            # normalise loss based on accumulation steps (simulate batching)
            D_loss_norm = D_loss / accumulation_steps
            # backpropagate
            D_loss_norm.backward()

            # train generator
            D_fake, fake_features = discriminator(input_img, generated_img)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))  # BCE for synth images
            l1_intensity_aware_loss = custom_loss(generated_img, target_img) * loss_lambda
            G_loss = G_fake_loss + l1_intensity_aware_loss
            # backpropagate and update generator weights
            G_loss_norm = G_loss / accumulation_steps
            G_loss_norm.backward()

            # update progress bar
            data_loader_tqdm.set_postfix({
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'Loss': l1_intensity_aware_loss.item()
            })

            # save metrics
            metrics['disc_losses'].append(D_loss.item())
            metrics['gen_losses'].append(G_loss.item())
            metrics['custom_loss_losses'].append(l1_intensity_aware_loss.item())

            # update weights with accumulated gradients
            if (idx + 1) % accumulation_steps == 0:
                optim_discriminator.step()
                optim_generator.step()
                optim_discriminator.zero_grad()
                optim_generator.zero_grad()

            # just save images from the first batch
            if idx == 0:
                batch_size = input_img.size(0)
                for batch_idx in range(batch_size):
                    img_name = img_names[batch_idx]

                    # crop and save input image
                    input_cropped = ut.remove_padding(input_img[batch_idx], original_size,
                                                      padding_coords, is_target=False)
                    input_path = os.path.join(run_name, f'e{epoch}_b{idx}_i_{img_name}')
                    ut.save_tensor_as_img(input_cropped, input_path)

                    # crop and save target image
                    target_cropped = ut.remove_padding(target_img[batch_idx], original_size,
                                                       padding_coords, is_target=True)
                    target_path = os.path.join(run_name, f'e{epoch}_b{idx}_t_{img_name}')
                    ut.save_tensor_as_img(target_cropped, target_path)

                    generated_cropped = ut.remove_padding(generated_img[batch_idx], original_size,
                                                          padding_coords, is_target=False)
                    generated_path = os.path.join(run_name, f'e{epoch}_b{idx}_g_{img_name}')
                    ut.save_tensor_as_img(generated_cropped, generated_path)

        # validation step
        generator.eval()
        val_psnr = 0
        val_ssim = 0
        num_val_batches = 0

        with torch.no_grad():
            # get a batch of validation data
            val_dir = os.path.join(run_name, 'validation')
            os.makedirs(val_dir, exist_ok=True)
            for val_idx, (val_input, val_target,
                          val_size, val_padding_coords, val_name) in enumerate(validation_loader):
                val_input, val_target = val_input.to(device), val_target.to(device)
                val_generated = generator(val_input)

                for batch_idx in range(val_input.size(0)):
                    dl = f'e{epoch}_b{batch_idx}'  # file ID
                    # crop images
                    val_target_cropped = ut.remove_padding(val_target[batch_idx], val_size,
                                                           val_padding_coords, is_target=True)
                    val_gen_cropped = ut.remove_padding(val_generated[batch_idx], val_size,
                                                        val_padding_coords, is_target=False)

                    val_psnr += psnr(val_gen_cropped.unsqueeze(0), val_target_cropped.unsqueeze(0))
                    val_ssim += ssim(val_gen_cropped.unsqueeze(0), val_target_cropped.unsqueeze(0))

                if val_idx == 0:
                    # just save images from the first batch
                    val_target_path = os.path.join(val_dir, f'{dl}_t_{str(val_name[batch_idx])}')
                    ut.save_tensor_as_img(val_target_cropped, val_target_path)

                    val_gen_path = os.path.join(val_dir, f'{dl}_g_{str(val_name[batch_idx])}')
                    ut.save_tensor_as_img(val_gen_cropped, val_gen_path)
                num_val_batches += 1

        # store average psnr and ssim
        avg_val_psnr = val_psnr / num_val_batches
        avg_val_ssim = val_ssim / num_val_batches
        metrics['avg_psnrs'].append(avg_val_psnr.item())
        metrics['avg_ssims'].append(avg_val_ssim.item())

        print(f'Validation PSNR: {avg_val_psnr:.4f}')  # 20-40 is best
        print(f'Validation SSIM: {avg_val_ssim:.4f}')  # ranges between -1 to 1
        print()

        # final update if number of batches is not divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            optim_discriminator.step()
            optim_generator.step()
            optim_discriminator.zero_grad()
            optim_generator.zero_grad()

    # save the model
    model_save_path = os.path.join(run_name, 'model')
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(model_save_path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(model_save_path, 'discriminator.pth'))

    # make directory for metrics captured during training
    metrics_path = os.path.join(run_name, 'metrics')
    os.makedirs(metrics_path, exist_ok=True)

    # plot and save discriminator loss
    disc_path = os.path.join(metrics_path, 'disc_loss')
    ut.save_figure(metrics['disc_losses'],
                   title='Discriminator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=disc_path)

    # plot and save generator loss
    gen_path = os.path.join(metrics_path, 'gen_loss')
    ut.save_figure(metrics['gen_losses'],
                   title='Generator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=gen_path)

    # compare discriminator and generator loss
    disc_gen_path = os.path.join(metrics_path, 'disc_gen_loss')
    ut.save_figure(metrics['disc_losses'],
                   metrics['gen_losses'],
                   title='Discriminator and Generator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=disc_gen_path)

    # plot and save L1 loss
    l1_path = os.path.join(metrics_path, 'l1_loss')
    ut.save_figure(metrics['custom_loss_losses'],
                   title='L1 Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=l1_path)

    psnr_path = os.path.join(metrics_path, 'psnr')
    ut.save_figure(metrics['avg_psnrs'],
                   title='Validation PSNR',
                   xlabel='Epoch',
                   ylabel='PSNR',
                   save_path=psnr_path)

    ssim_path = os.path.join(metrics_path, 'ssim')
    ut.save_figure(metrics['avg_ssims'],
                   title='Validation SSIM',
                   xlabel='Epoch',
                   ylabel='SSIM',
                   save_path=ssim_path)

    # save all metrics for inspection and clarity
    with open(os.path.join(metrics_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
