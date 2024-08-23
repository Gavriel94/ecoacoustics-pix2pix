"""
Evaluates the model on test data.

Creates and populates a folder 'evaluate' in the dataset folder with
- Audio recomposed from synthesised spectrograms, using their target's magnitude and phase.
- Images of generated and associative real spectrograms saved during testing.
- Graphs with average PSNR and SSIM data.
- A table of acoustic indices that compare generated and target audio files.
- A table of birds as predicted by BirdNet on generated and target audio files.
"""
import os

import torch
from torch.utils.data import DataLoader

import config
import pix2pix.model_utils as m_utils
from Acoustic_Indices.main_compute_indices_from_dir import compute_indices
import utilities as utils
from pix2pix.dataset_eval import Pix2PixEvaluationDataset
from pix2pix.generator_transpose_conv import Generator as GenRegTranspose
from pix2pix.generator_regular_conv import Generator as GenRegConv
from pix2pix.test_cGAN import test_model
from pix2pix.birdnet_eval import birdnet_analysis


# * ensure hyperparameters match ones used during training

def main():
    # which training run to evaluate
    run_num = 8  # ! increment this !
    eval_dir = os.path.join(config.DATASET_ROOT, 'evaluation', f'run_{run_num}')
    os.makedirs(eval_dir, exist_ok=False)

    # ensure no metadata files exist
    utils.remove_hidden_files(config.RAW_DATA_ROOT)
    utils.remove_hidden_files(config.DATASET_ROOT)

    test_dataset = Pix2PixEvaluationDataset(dataset=os.path.join(config.DATASET_ROOT, 'test'),
                                      use_correlated=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS,
                             shuffle=True,
                             collate_fn=m_utils.eval_collate)

    generator_path = os.path.join(config.DATASET_ROOT, 'train_runs',
                                  f'run_{run_num}', 'model', 'generator.pth')
    gen = GenRegConv(in_ch=1, features=64).to(config.DEVICE)
    gen.load_state_dict(torch.load(generator_path, weights_only=False))

    # test model
    test_model(test_loader, gen, run_num)

    # compute acoustic indices
    gen_audio_root = os.path.join(eval_dir, 'generated_audio')
    print(eval_dir)
    print(gen_audio_root)
    gen_audio = os.listdir(gen_audio_root)
    summaries_root = os.path.join(config.RAW_DATA_ROOT, 'full_summaries')

    # get full paths to generated audio
    full_gen_paths = [os.path.join(eval_dir, 'generated_audio', name)
                      for name in gen_audio]
    # compute indices
    compute_indices(full_gen_paths, eval_dir, config.TARGET_MIC_DELIM, True)

    # get full paths to target audio
    full_target_paths = utils.get_raw_audio(gen_audio)
    # compute acoustic indices
    compute_indices(full_target_paths, eval_dir, config.TARGET_MIC_DELIM, False)

    # compare generated and raw audio with BirdNet
    birdnet_analysis(summaries_root, gen_audio_root, run_num, random_samples=-1)


if __name__ == '__main__':
    main()
