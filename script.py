import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_dataset
from cGAN.dataset import SpectrogramDataset
import cGAN.utilities as utils
from cGAN import config as cfg
from cGAN.train import train_model
from cGAN.discriminator import Discriminator
from cGAN.generator import Generator


DATA_ROOT = 'raw_data/'
DATASET_ROOT = 'data/'
DATASET_PATH = f'{DATASET_ROOT}dataset'


def main():
    if not os.path.isdir(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        print('Creating a new dataset in \'data/\'')
        create_dataset(data_root='raw_data/',
                       dataset_root='data/',
                       analysis=False,
                       matched_summaries=cfg.MATCHED_SUMMARIES,
                       copied_recordings=cfg.COPIED_RECORDINGS,
                       spectrogram_paths=cfg.SPECTROGRAM_PATHS,
                       verbose=True)

    # initialise DataLoaders
    files = utils.get_files(DATASET_PATH, include_correlated=False)
    train, val, test = utils.split_data(files, 0.8)

    train_dataset = SpectrogramDataset(train, augment=False)
    val_dataset = SpectrogramDataset(val, augment=False)
    test_dataset = SpectrogramDataset(test, augment=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              num_workers=cfg.NUM_WORKERS,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.BATCH_SIZE,
                            num_workers=cfg.NUM_WORKERS,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.BATCH_SIZE,
                             num_workers=cfg.NUM_WORKERS,
                             shuffle=False)

    disc = Discriminator(in_ch=1).to(cfg.DEVICE)
    gen = Generator(in_ch=1, features=64).to(cfg.DEVICE)
    # setting betas reduce momentum (used in the paper)
    optim_disc = optim.Adam(disc.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    disc_loss, gen_loss, l1_loss = train_model(disc, gen, train_loader,
                                               optim_disc, optim_gen, L1_LOSS,
                                               BCE)
    utils.plot_loss(disc_loss, gen_loss, l1_loss)


if __name__ == '__main__':
    main()
