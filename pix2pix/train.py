import torch
from tqdm import tqdm
import os
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from . import utilities as ut


def train(discriminator, generator, train_loader, validation_loader, optim_discriminator, optim_generator, L1_loss, l1_lambda, bce_logits, num_epochs, device, save_dir: str, accumulation_steps: int = 1, display_epoch: int = 5):
    os.makedirs(f'{save_dir}/evaluation', exist_ok=True)
    run_name = f"{save_dir}/evaluation/run_{len(os.listdir(f'{save_dir}/evaluation')) + 1}"
    os.makedirs(run_name, exist_ok=True)
    disc_losses, gen_losses, l1_losses = [], [], []
    val_psnr_scores, val_ssim_scores = [], []

    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        discriminator.train()
        generator.train()
        data_loader_tqdm = tqdm(train_loader, leave=True)
        for idx, (input_img, target_img, original_size, padding_coords, img_names) in enumerate(data_loader_tqdm):
            input_img, target_img = input_img.to(device), target_img.to(device)

            # train discriminator
            generated_img = generator(input_img)
            D_real, real_features = discriminator(input_img, target_img)
            D_fake, fake_features = discriminator(input_img, generated_img.detach())
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
            D_fake, fake_features = discriminator(input_img, generated_img)  # disc output for synth pair (input, generated)
            G_fake_loss = bce_logits(D_fake, torch.ones_like(D_fake))  # BCE for synth images (gen tries to fool dis, should be close to 1)
            L1 = L1_loss(generated_img, target_img) * l1_lambda # L1 loss between synth and target images encourage visual similarity
            G_loss = G_fake_loss + L1  # generator loss is a combination of adverserial (G_fake_loss) and L1 loss
            # backpropagate and update generator weights
            G_loss_norm = G_loss / accumulation_steps
            G_loss_norm.backward()

            # update progress bar
            data_loader_tqdm.set_postfix({
                'D_loss': D_loss.item(),
                'G_loss': G_loss.item(),
                'L1_loss': L1.item()
            })

            # save metrics
            disc_losses.append(D_loss.item())
            gen_losses.append(G_loss.item())
            l1_losses.append(L1.item())

            # update weights with accumulated gradients
            if (idx + 1) % accumulation_steps == 0:
                optim_discriminator.step()
                optim_generator.step()
                optim_discriminator.zero_grad()
                optim_generator.zero_grad()

            # save images to disk for inspection
            if idx % display_epoch == 0:
                batch_size = input_img.size(0)
                for i in range(batch_size):
                    # os.makedirs(save_path, exist_ok=True)
                    batch_size = input_img.size(0)
                    for i in range(batch_size):
                        img_name = img_names[i]

                    # crop and save input image
                    input_cropped = ut.remove_padding(input_img[i], original_size,
                                                      padding_coords, is_target=False)
                    input_path = os.path.join(run_name, f'e{epoch}_b{idx}_i_{img_name}')
                    ut.save_tensor_as_img(input_cropped, input_path)

                    # crop and save target image
                    target_cropped = ut.remove_padding(target_img[i], original_size,
                                                       padding_coords, is_target=True)
                    target_path = os.path.join(run_name, f'e{epoch}_b{idx}_t_{img_name}')
                    ut.save_tensor_as_img(target_cropped, target_path)

                    generated_cropped = ut.remove_padding(generated_img[i], original_size,
                                                          padding_coords, is_target=False)
                    generated_path = os.path.join(run_name, f'e{epoch}_b{idx}_g_{img_name}')
                    ut.save_tensor_as_img(generated_cropped, generated_path)

        # validation step
        generator.eval()
        val_psnr = 0
        val_ssim = 0
        num_val_batches = 0

        with torch.no_grad():
            for idx, (val_input, val_target, val_size, val_padding_coords, val_names) in enumerate(validation_loader):
                val_input, val_target = val_input.to(device), val_target.to(device)
                val_generated = generator(val_input)

                val_psnr += psnr(val_generated, val_target)
                val_ssim += ssim(val_generated, val_target)
                num_val_batches += 1

            if idx % display_epoch == 0:
                val_dir = os.path.join(run_name, 'validation')
                os.makedirs(val_dir, exist_ok=True)
                for i in range(min(3, val_input.size(0))):
                    val_name = str(val_names[i])
                    ut.save_tensor_as_img(ut.remove_padding(val_input[i], val_size,
                                                            val_padding_coords, is_target=False),
                                          os.path.join(val_dir, f'e{epoch}_b{idx}_i_{val_name}'))

                    ut.save_tensor_as_img(ut.remove_padding(val_target[i], val_size,
                                                            val_padding_coords, is_target=True),
                                          os.path.join(val_dir, f'e{epoch}_b{idx}_t_{val_name}'))

                    ut.save_tensor_as_img(ut.remove_padding(val_generated[i], val_size,
                                                            val_padding_coords, is_target=False),
                                          os.path.join(val_dir, f'e{epoch}_b{idx}_g_{val_name}'))

        avg_val_psnr = val_psnr / num_val_batches
        avg_val_ssim = val_ssim / num_val_batches
        val_psnr_scores.append(avg_val_psnr.item())
        val_ssim_scores.append(avg_val_ssim.item())

        print(f'Validation PSNR: {avg_val_psnr:.4f}')  # 20-40 is best
        print(f'Validation SSIM: {avg_val_ssim:.4f}')  # ranges between -1 to 1 with 1 being best
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
