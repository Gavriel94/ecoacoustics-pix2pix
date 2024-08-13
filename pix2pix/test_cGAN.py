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


def test_model(dataloader, generator, device, save_dir, run_num):
    eval_dir = os.path.join(save_dir, 'evaluate')
    os.makedirs(eval_dir, exist_ok=True)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(config.DEVICE)
    ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)

    metrics = {
        'avg_psnrs': [],
        'avg_ssims': [],
    }
    avg_psnr = 0
    avg_ssim = 0

    generator.eval()
    print(f'Evaluating run {run_num}')
    with torch.no_grad():
        dataloader_tqdm = tqdm(dataloader)
        for idx, (input_img, target_img, original_size,
                  padding_coords, img_names, param_dicts, raw_audio) in enumerate(dataloader_tqdm):
            input_img, target_img = input_img.to(device), target_img.to(device)
            gen_spec = generator(input_img)

            num_batches = 0
            for batch_idx in range(input_img.size(0)):
                img_dir = os.path.join(eval_dir, 'images')
                os.makedirs(img_dir, exist_ok=True)
                # crop images
                gen_cropped = utils.remove_padding_from_tensor(gen_spec[batch_idx],
                                                               original_size,
                                                               padding_coords,
                                                               is_target=False)
                target_cropped = utils.remove_padding_from_tensor(target_img[batch_idx],
                                                                  original_size,
                                                                  padding_coords,
                                                                  is_target=True)
                # compute psnr and ssim
                avg_psnr += psnr(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))
                avg_ssim += ssim(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))

                # save target tensor
                target_save = os.path.join(img_dir, f'b{idx}i{batch_idx}_t_{img_names[batch_idx]}')
                utils.save_tensor_as_img(target_cropped, target_save)

                # convert generated tensor to image
                gen_img = utils.convert_tensor_to_img(gen_cropped)
                img_save_path = os.path.join(img_dir,
                                             f'b{idx}i{batch_idx}_g_{img_names[batch_idx]}')
                # save using PIL and keep generated img in memory
                gen_img.save(img_save_path)

                # create audio from spectrograms and save
                audio_dir = os.path.join(eval_dir, 'generated audio')
                os.makedirs(audio_dir, exist_ok=True)
                save_path = os.path.join(audio_dir, img_names[batch_idx])  # filename has .png ext
                try:
                    utils.spectrogram_to_audio(gen_img, param_dicts[batch_idx], save_path, 48000)
                except json.decoder.JSONDecodeError as e:
                    print(f'Error opening {param_dicts[batch_idx]}, {str(e)}')
            num_batches += 1

            avg_psnr = avg_psnr / num_batches
            avg_ssim = avg_ssim / num_batches
            metrics['avg_psnrs'].append(avg_psnr.item())
            metrics['avg_ssims'].append(avg_ssim.item())

        print('metrics[\'avg_psnrs\']', metrics['avg_psnrs'])
        print('metrics[\'avg_ssims\']', metrics['avg_ssims'])

        graph_dir = os.path.join(eval_dir, 'graphs')
        os.makedirs(graph_dir, exist_ok=True)

        psnr_path = os.path.join(graph_dir, 'psnr')
        utils.create_line_graph(metrics['avg_psnrs'],
                                title='Average PSNR',
                                xlabel='Batch',
                                ylabel='PSNR',
                                save_path=psnr_path)

        ssim_path = os.path.join(graph_dir, 'ssim')
        utils.create_line_graph(metrics['avg_ssims'],
                                title='Average SSIM',
                                xlabel='Batch',
                                ylabel='SSIM',
                                save_path=ssim_path)
    return metrics
