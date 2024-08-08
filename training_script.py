import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from pix2pix.ds import Pix2PixDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from dataset import create_dataset
from pix2pix.train import train_model
import pix2pix.utilities as utils
# import logging


# def setup_logging():
#     logging.basicConfig(format='%(asctime)s:%(message)s',
#                         level=logging.INFO,
#                         datefmt='%m/%d/%Y %I:%M:%S %p')
#     logging.getLogger('PIL').setLevel(logging.WARNING)

RAW_DATA = 'raw_data_test/'
DATASET = 'data/'
DATASET_PATH = os.path.join(DATASET, 'dataset')

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


def main():
    create_new_dataset = False

    if create_new_dataset:
        print(f'Creating {DATASET_PATH}\n')
        create_dataset(data_root=RAW_DATA,
                       dataset_root=DATASET,
                       analysis=False,
                       matched_summaries=MATCHED_SUMMARIES,
                       copied_recordings=COPIED_RECORDINGS,
                       spectrogram_paths=SPECTROGRAM_PATHS,
                       verbose=True)

    # get files from dataset
    files = utils.get_files(DATASET_PATH, include_correlated=False)

    # create train, val, test sets
    train, _, _ = utils.split_data(files, 0.8)

    train_dataset = Pix2PixDataset(train, augment=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              collate_fn=utils.custom_collate)

    disc = Discriminator(in_ch=1).to(DEVICE)
    gen = Generator(in_ch=1, features=64).to(DEVICE)
    # setting betas reduce momentum (used in the paper)
    optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    disc_loss, gen_loss, l1_loss = train_model(disc, gen, train_loader,
                                               optim_disc, optim_gen, L1_LOSS, L1_LAMBDA,
                                               BCE, NUM_EPOCHS, DEVICE)
    utils.plot_loss(disc_loss, gen_loss, l1_loss)


if __name__ == '__main__':
    main()
