"""
Evaluates the model on test data.

Creates an `evaluate` folder in the dataset
"""
import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import (PeakSignalNoiseRatio,
                                StructuralSimilarityIndexMeasure)
from tqdm import tqdm

import config
from Acoustic_Indices.main_compute_indices_from_dir import compute_indices
from pix2pix import utilities as utils
from pix2pix.dataset_eval import Pix2PixEvalDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator


def test_model(dataloader, generator, device, save_dir, run_num):
    eval_dir = os.path.join(save_dir, 'evaluate')
    os.makedirs(eval_dir, exist_ok=True)

    psnr = PeakSignalNoiseRatio().to(config.DEVICE)
    ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)

    metrics = {
        'avg_psnr': [],
        'avg_ssim': [],
        'audio_comparisons': []
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
                gen_cropped = utils.remove_padding(gen_spec[batch_idx], original_size,
                                                   padding_coords, is_target=False)
                target_cropped = utils.remove_padding(target_img[batch_idx], original_size,
                                                      padding_coords, is_target=True)

                avg_psnr += psnr(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))
                avg_ssim += ssim(gen_cropped.unsqueeze(0), target_cropped.unsqueeze(0))

                target_save = os.path.join(img_dir, f'b{idx}i{batch_idx}_t_{img_names[batch_idx]}')
                utils.save_tensor_as_img(target_cropped, target_save)

                gen_img = utils.convert_tensor_to_img(gen_cropped)
                img_save_path = os.path.join(img_dir, f'b{idx}i{batch_idx}_g_{img_names[batch_idx]}')
                print(img_save_path)
                gen_img.save(img_save_path)  # save using PIL to keep gen_img in memory

                # create audio from spectrograms and save
                audio_dir = os.path.join(eval_dir, 'audio')
                os.makedirs(audio_dir, exist_ok=True)
                save_path = os.path.join(audio_dir, img_names[batch_idx])  # filename with .png ext
                try:
                    gen_audio = utils.spectrogram_to_audio(gen_img, param_dicts[batch_idx], save_path, 48000)
                except json.decoder.JSONDecodeError as e:
                    print(f'Error opening {param_dicts[batch_idx]}, {str(e)}')
            num_batches += 1

        avg_psnr = avg_psnr / num_batches
        avg_ssim = avg_ssim / num_batches
        metrics['avg_psnr'].append(avg_psnr)
        metrics['avg_ssim'].append(avg_ssim)
    return metrics


def main():
    raw_data = config.RAW_DATA_ROOT
    data = config.DATASET_ROOT

    target_mic = 'SM4'
    mic_delim = '-4'
    test_dataset = Pix2PixEvalDataset(dataset=os.path.join(config.DATASET_ROOT, 'test'),
                                      use_correlated=False,
                                      mic2_name=target_mic,
                                      mic2_delim=mic_delim,
                                      raw_data_root=raw_data)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS,
                             shuffle=True,
                             collate_fn=utils.eval_collate)

    # get saved model
    run_num = 1
    generator_path = os.path.join(data, 'runs', f'run_{run_num}', 'model', 'generator.pth')
    discrimin_path = os.path.join(data, 'runs', f'run_{run_num}', 'model', 'discriminator.pth')

    gen = Generator(in_ch=1, features=64).to(config.DEVICE)
    dis = Discriminator(in_ch=1).to(config.DEVICE)

    gen.load_state_dict(torch.load(generator_path, weights_only=False))
    dis.load_state_dict(torch.load(discrimin_path, weights_only=False))

    test_model(test_loader, gen, config.DEVICE, data, run_num)

    if os.path.exists(os.path.join(data, 'evaluate', 'acoustic_indices.csv')):
        os.remove(os.path.join(data, 'evaluate', 'acoustic_indices.csv'))

    gen_audio = os.listdir(os.path.join(data, 'evaluate', 'audio'))
    full_gen_paths = [os.path.join(data, 'evaluate', 'audio', name) for name in os.listdir(os.path.join(data, 'evaluate', 'audio'))]
    compute_indices(full_gen_paths, os.path.join(data, 'evaluate'), mic_delim, True)

    full_target_paths = utils.get_raw_audio(gen_audio, raw_data, 'SM4', '-4')
    compute_indices(full_target_paths, os.path.join(data, 'evaluate'), mic_delim, False)


if __name__ == '__main__':
    main()
