"""
Define where raw data is stored and where the dataset and evaluation outputs are stored.
Define cGAN hyperparameters.
"""

import pix2pix.utilities as utils

DATASET_ROOT = 'data_test/'
RAW_DATA_ROOT = 'raw_data_test/'

DEVICE = utils.set_device('mps')
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 2
L1_LAMBDA = 200  # how much weight is given to the combined L1/Intensity loss
DISPLAY_EPOCH = 2  # epoch where progress information is displayed

# number of steps gradients accumulate before updating weights
ACCUMULATION_STEPS = 4  # (emulated batch size == ACCUMULATION_STEPS * BATCH_SIZE)
