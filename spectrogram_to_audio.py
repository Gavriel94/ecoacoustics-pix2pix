"""
Script to turn a spectrogram into audio using a Pix2Pix utility function.
"""

import os
from pix2pix import utilities as utils


def main():
    spectrogram_path = 'data/spectrograms/PLI1_20240316_115600.png'
    filename = spectrogram_path.split('/')[2].replace('.png', '.wav')

    output_path = os.path.join('data', 'spectrograms', 'spectrogram_to_audio')
    os.makedirs(output_path, exist_ok=True)

    overwrite = True
    if not overwrite:
        if os.path.exists(os.path.join(output_path, filename)):
            raise FileExistsError

    utils.spectrogram_to_audio(spectrogram_path, output_path, sample_rate=48000)


if __name__ == '__main__':
    main()
