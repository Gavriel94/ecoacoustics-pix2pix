"""
Declare variables to shortcut parts of the dataset creation process and
alter the cGAN configuration.
"""

import cGAN.utilities as utils


# Creating the dataset can be a lengthy process.
# By saving these variables, it's possible to shortcut some of the process
# Ensure you've set these to None if the dataset has not yet been created.
MATCHED_SUMMARIES = [['data/2024_03/PLI1/summary.csv',
                      'data/2024_03/PLI2/summary.csv',
                      'data/2024_03/PLI3/summary.csv'],
                     ['data/2023_11/PLI2/summary.csv',
                      'data/2023_11/PLI3/summary.csv']]

COPIED_RECORDINGS = [['data/2024_03/PLI1',
                      'data/2024_03/PLI2',
                      'data/2024_03/PLI3'],
                     ['data/2023_11/PLI2',
                      'data/2023_11/PLI3']]

SPECTROGRAM_PATHS = [['data/spectrograms/2024_03/PLI2',
                      'data/spectrograms/2024_03/PLI3',
                      'data/spectrograms/2024_03/PLI1'],
                     ['data/spectrograms/2023_11/PLI2',
                      'data/spectrograms/2023_11/PLI3']]

# Model config parameters
DEVICE = utils.set_device('mps')
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 6
NUM_EPOCHS = 3
L1_LAMBDA = 100
