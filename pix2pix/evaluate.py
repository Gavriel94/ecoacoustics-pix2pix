"""
Test the spectrogram translation conditional Generative Adversial Network.
"""

import os
import json
from tqdm import tqdm
import config
from . import model_utils as utils

import torch
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)


def test_model(dataloader, generator, run_num):
    """
    Evaluate the model using test data and calculate PSNR and SSIM for generated
    and target images.

    Args:
        dataloader (DataLoader): Test data.
        generator (Generator): Generator used in the cGAN.
        run_num (int): Which run is being evaluated, used as label for file.
    """
    eval_dir = os.path.join(config.DATASET_ROOT, 'evaluation', f'run_{run_num}')

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(config.DEVICE)
    ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)

    total_psnr = 0
    total_ssim = 0
    total_images = 0

    generator.eval()
    print(f'Evaluating run {run_num}')
    with torch.no_grad():
        for batch_idx, (input_img, target_img, original_size, padding_coords,
                        img_names, param_dicts, raw_audio) in enumerate(tqdm(dataloader)):
            input_img, target_img = input_img.to(config.DEVICE), target_img.to(config.DEVICE)
            gen_spec = generator(input_img)

            for i in range(input_img.size(0)):
                img_dir = os.path.join(eval_dir, 'images')
                os.makedirs(img_dir, exist_ok=True)
                # crop images
                gen_cropped = utils.remove_padding_from_tensor(gen_spec[i],
                                                               original_size,
                                                               padding_coords,
                                                               is_target=False)
                target_cropped = utils.remove_padding_from_tensor(target_img[i],
                                                                  original_size,
                                                                  padding_coords,
                                                                  is_target=True)
                # compute psnr and ssim
                total_psnr += psnr(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))
                total_ssim += ssim(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))
                total_images += 1

                # save target tensor
                target_save = os.path.join(img_dir, f'b{batch_idx}i{i}_t_{img_names[i]}')
                utils.save_tensor_as_img(target_cropped, target_save)

                # convert generated tensor to image
                gen_img = utils.convert_tensor_to_img(gen_cropped)
                img_save_path = os.path.join(img_dir, f'b{batch_idx}i{i}_g_{img_names[i]}')
                # save using PIL and keep generated img in memory
                gen_img.save(img_save_path)

                # create audio from spectrograms and save
                audio_dir = os.path.join(eval_dir, 'generated_audio')
                os.makedirs(audio_dir, exist_ok=True)
                save_path = os.path.join(audio_dir, img_names[i])  # filename has .png ext
                try:
                    utils.spectrogram_to_audio(gen_img, param_dicts[i], save_path)
                except json.decoder.JSONDecodeError as e:
                    print(f'Error opening {param_dicts[i]}, {str(e)}')
                print(f'Generated audio from {img_names[i]}')

    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images

    # Save metrics to a JSON file
    metrics = {
        'avg_psnr': avg_psnr.item(),
        'avg_ssim': avg_ssim.item(),
        'total_images': total_images
    }
    with open(os.path.join(eval_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print(f'Total images processed: {total_images}')

    return metrics
