"""
Script to turn spectrograms from the test set into audio using a Pix2Pix utility function.
Used to evaluate the cGANs synthesised spectrograms with unseen data.
"""
import os

from pix2pix import utilities as utils
import config


def main():
    raw_data = config.RAW_DATA_ROOT
    data = config.DATASET_ROOT

    target_mic = 'SM4'
    target_mic_delim = '-4'

    audio_paths, param_paths = utils.get_test_sample(dataset_root=data,
                                                     raw_data_root=raw_data,
                                                     mic2_name=target_mic,
                                                     mic2_delim=target_mic_delim,
                                                     num_samples=3)

    # perform inference with model
    for audio in audio_paths:
        for params in param_paths:
            if os.path.basename(audio).replace('.wav', '') == params.replace('.json.gz', ''):
                utils.spectrogram_to_audio(audio, params, config.DATASET_ROOT, 'tmp/e', 48000)


if __name__ == '__main__':
    main()
