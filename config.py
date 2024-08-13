"""
Define cGAN hyperparameters, raw data and dataset paths.
"""

from pix2pix.model_utils import set_device

DATASET_ROOT = 'data_test/'
RAW_DATA_ROOT = 'raw_data_test/'

DEVICE = set_device('mps')

LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 0
NUM_EPOCHS = 2

# contribution of intensity-awareness loss to total Generator loss
L1_LAMBDA = 400

# number of steps gradients accumulate before updating weights
ACCUMULATION_STEPS = 4  # emulates batch of size (ACCUMULATION_STEPS * BATCH_SIZE)
