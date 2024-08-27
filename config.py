"""
Define cGAN hyperparameters, raw data and dataset paths.
"""
from pix2pix.l1_intensity_loss import Pix2PixL1IntensityLoss
from pix2pix.l1_loss import Pix2PixL1Loss
from pix2pix.model_utils import set_device

DATASET_ROOT = 'data/'
RAW_DATA_ROOT = 'raw_data/'

"""
cGAN variables
"""
# Tunable parameters
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
L1_LAMBDA = 200  # contribution of custom loss to total Generator loss

CUSTOM_LOSS = Pix2PixL1IntensityLoss(alpha=0.8)
# CUSTOM_LOSS = Pix2PixL1IntensityLoss()

# this emulates a batch of size (ACCUMULATION_STEPS * BATCH_SIZE)
ACCUMULATION_STEPS = 4  # number of steps before parameters update

"""
Dataset variables
"""
# what % of data should be in the train set
TRAIN_PCT = 0.8
DEVICE = set_device('mps')
BATCH_SIZE = 1
NUM_WORKERS = 0

# name of microphone providing input audio
INPUT_MIC_NAME = 'SMMicro'
# name of audio providing target audio
TARGET_MIC_NAME = 'SM4'
# delimiter used to differentiate between input and target file names
TARGET_MIC_DELIM = '-4'
