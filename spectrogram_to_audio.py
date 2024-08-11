"""
Script to turn spectrograms from the test set into audio using a Pix2Pix utility function.
Used to evaluate the cGANs synthesised spectrograms with unseen data.
"""
from pix2pix import utilities as utils
import config


def main():
    raw_data = config.RAW_DATA_ROOT
    data = config.DATASET_ROOT

    target_mic = 'SM4'
    target_mic_delim = '-4'

    test_samples = utils.get_test_sample(dataset_root=data,
                                         raw_data_root=raw_data,
                                         mic2_name=target_mic,
                                         mic2_delim=target_mic_delim,
                                         num_samples=3)
    print(test_samples)

    # output_path = os.path.join('data', 'spectrograms', 'spectrogram_to_audio')
    # os.makedirs(output_path, exist_ok=True)

    # overwrite = True
    # if not overwrite:
    #     if os.path.exists(os.path.join(output_path, filename)):
    #         raise FileExistsError

    # utils.spectrogram_to_audio(spectrogram_path, output_path, sample_rate=48000)


if __name__ == '__main__':
    main()
