"""
Define paths for files used for creating the dataset and alter model hyperparameters.
"""

import pix2pix.utilities as utils
import os
# import logging

DATA_ROOT = 'raw_data_test/'
DATASET_ROOT = 'data/'
DATASET_PATH = os.path.join(DATASET_ROOT, 'dataset')

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
NUM_EPOCHS = 2
L1_LAMBDA = 100


# def setup_logging():
#     logging.basicConfig(format='%(asctime)s:%(message)s',
#                         level=logging.INFO,
#                         datefmt='%m/%d/%Y %I:%M:%S %p')
#     logging.getLogger('PIL').setLevel(logging.WARNING)
