"""
Train the spectrogram translation conditional Generative Adversial Network.
"""

import os

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
          accumulation_steps, view_val_epoch: int = 5):
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
        view_val_epoch (int, optional): View attempts on validation data. Defaults to 5.
    """
    os.makedirs(f'{save_dir}/evaluation', exist_ok=True)
    num_runs = len(os.listdir(f'{save_dir}/evaluation'))
    run_name = f"{save_dir}/evaluation/run_{num_runs + 1}"
    os.makedirs(run_name, exist_ok=True)

    disc_losses, gen_losses, l1_losses = [], [], []
    val_psnr_scores, val_ssim_scores = [], []
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    print(f'Run {num_runs + 1}')
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
            disc_losses.append(D_loss.item())
            gen_losses.append(G_loss.item())
            l1_losses.append(l1_intensity_aware_loss.item())

            # update weights with accumulated gradients
            if (idx + 1) % accumulation_steps == 0:
                optim_discriminator.step()
                optim_generator.step()
                optim_discriminator.zero_grad()
                optim_generator.zero_grad()

        # save images to disk for inspection
        if idx % view_val_epoch == 0:
            batch_size = input_img.size(0)
            for val_idx in range(batch_size):
                # os.makedirs(save_path, exist_ok=True)
                batch_size = input_img.size(0)
                for val_idx in range(batch_size):
                    img_name = img_names[val_idx]

                # crop and save input image
                input_cropped = ut.remove_padding(input_img[val_idx], original_size,
                                                  padding_coords, is_target=False)
                input_path = os.path.join(run_name, f'e{epoch}_b{idx}_i_{img_name}')
                ut.save_tensor_as_img(input_cropped, input_path)

                # crop and save target image
                target_cropped = ut.remove_padding(target_img[val_idx], original_size,
                                                   padding_coords, is_target=True)
                target_path = os.path.join(run_name, f'e{epoch}_b{idx}_t_{img_name}')
                ut.save_tensor_as_img(target_cropped, target_path)

                generated_cropped = ut.remove_padding(generated_img[val_idx], original_size,
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
            for val_idx, (val_input, val_target,
                          val_size, val_padding_coords, val_name) in enumerate(validation_loader):
                val_input, val_target = val_input.to(device), val_target.to(device)
                val_generated = generator(val_input)

                # get psnr and ssim by comparing generated and target images
                val_psnr += psnr(val_generated, val_target)
                val_ssim += ssim(val_generated, val_target)
                num_val_batches += 1

            if idx % view_val_epoch == 0:
                val_dir = os.path.join(run_name, 'validation')
                os.makedirs(val_dir, exist_ok=True)
                for batch_idx in range(min(3, val_input.size(0))):
                    dl = f'e{epoch}_b{batch_idx}'  # file ID

                    # uncomment to save input images as well
                    # val_input_cropped = ut.remove_padding(val_input[batch_idx], val_size,
                    #                                       val_padding_coords, is_target=False)
                    # val_input_path = os.path.join(val_dir, f'{dl}_i_{str(val_name[batch_idx])}')
                    # ut.save_tensor_as_img(val_input_cropped, val_input_path)

                    # crop and save target validation image
                    val_target_cropped = ut.remove_padding(val_target[batch_idx], val_size,
                                                           val_padding_coords, is_target=True)
                    val_target_path = os.path.join(val_dir, f'{dl}_t_{str(val_name[batch_idx])}')
                    ut.save_tensor_as_img(val_target_cropped, val_target_path)

                    # crop and save generated validation image
                    val_gen_cropped = ut.remove_padding(val_generated[batch_idx], val_size,
                                                        val_padding_coords, is_target=False)
                    val_gen_path = os.path.join(val_dir, f'{dl}_g_{str(val_name[batch_idx])}')
                    ut.save_tensor_as_img(val_gen_cropped, val_gen_path)

        # store average psnr and ssim
        avg_val_psnr = val_psnr / num_val_batches
        avg_val_ssim = val_ssim / num_val_batches
        val_psnr_scores.append(avg_val_psnr.item())
        val_ssim_scores.append(avg_val_ssim.item())

        print(f'Validation PSNR: {avg_val_psnr:.4f}')  # 20-40 is best
        print(f'Validation SSIM: {avg_val_ssim:.4f}')  # ranges between -1 to 1
        print()

        # final update if number of batches is not divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            optim_discriminator.step()
            optim_generator.step()
            optim_discriminator.zero_grad()
            optim_generator.zero_grad()

    # make directory for metrics captured during training
    graphs_path = os.path.join(run_name, 'graphs')
    os.makedirs(graphs_path, exist_ok=True)

    # plot and save discriminator loss
    disc_path = os.path.join(graphs_path, 'disc_loss')
    ut.save_figure(disc_losses,
                   title='Discriminator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=disc_path)

    # plot and save generator loss
    gen_path = os.path.join(graphs_path, 'gen_loss')
    ut.save_figure(gen_losses,
                   title='Generator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=gen_path)

    # compare discriminator and generator loss
    disc_gen_path = os.path.join(graphs_path, 'disc_gen_loss')
    ut.save_figure(disc_losses,
                   gen_losses,
                   title='Discriminator and Generator Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=disc_gen_path)

    # plot and save L1 loss
    l1_path = os.path.join(graphs_path, 'l1_loss')
    ut.save_figure(l1_losses,
                   title='L1 Loss',
                   xlabel='Epoch',
                   ylabel='Loss',
                   save_path=l1_path)

    psnr_path = os.path.join(graphs_path, 'psnr')
    ut.save_figure(val_psnr_scores,
                   title='Validation PSNR',
                   xlabel='Epoch',
                   ylabel='PSNR',
                   save_path=psnr_path)

    ssim_path = os.path.join(graphs_path, 'ssim')
    ut.save_figure(val_ssim_scores,
                   title='Validation SSIM',
                   xlabel='Epoch',
                   ylabel='SSIM',
                   save_path=ssim_path)
