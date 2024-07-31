"""
Set parameters such as learning rate, batch size, device and more.
"""

import cGAN.utilities as utils

DEVICE = utils.set_device('mps')
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 2
L1_LAMBDA = 100
