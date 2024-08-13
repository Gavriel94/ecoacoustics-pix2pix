"""
Evaluates the model on test data.

Creates an `evaluate` folder in the dataset
"""
import os

import torch
from torch.utils.data import DataLoader

import config
from Acoustic_Indices.main_compute_indices_from_dir import compute_indices
import utilities as utils
from pix2pix.dataset_eval import Pix2PixEvalDataset
from pix2pix.generator import Generator
from pix2pix.test_cGAN import test_model
from tmp import birdnet_analysis


def main():
    raw_data = config.RAW_DATA_ROOT
    data = config.DATASET_ROOT

    # ensure no metadata files exist
    utils.remove_hidden_files(config.RAW_DATA_ROOT)
    utils.remove_hidden_files(config.DATASET_ROOT)

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
    gen = Generator(in_ch=1, features=64).to(config.DEVICE)
    gen.load_state_dict(torch.load(generator_path, weights_only=False))

    # test model
    test_model(test_loader, gen, config.DEVICE, data, run_num)

    # compute acoustic indices
    gen_audio_root = os.listdir(os.path.join(data, 'evaluate', 'audio'))
    # get full paths to generated .wav files
    full_gen_paths = [os.path.join(data, 'evaluate', 'audio', name)
                      for name in gen_audio_root]
    # compute indices
    compute_indices(full_gen_paths, os.path.join(data, 'evaluate'), mic_delim, True)

    # get full paths to target .wav files
    full_target_paths = utils.get_raw_audio(gen_audio_root, raw_data, 'SM4', '-4')
    # compute indices
    compute_indices(full_target_paths, os.path.join(data, 'evaluate'), mic_delim, False)

    # BirdNet analysis
    summaries_root = os.path.join(data, 'full_summaries')
    birdnet_analysis(summaries_root, gen_audio_root, num_samples=3)


if __name__ == '__main__':
    main()
