import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from pix2pix.dataset import Pix2PixDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from pix2pix.train import train_model
import pix2pix.utilities as utils

DATASET = 'data/'
DATASET_PATH = os.path.join(DATASET, 'dataset')


# Model config parameters
DEVICE = utils.set_device('mps')
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 6
NUM_EPOCHS = 2
L1_LAMBDA = 100


def main():

    train_dataset = Pix2PixDataset(dataset='data/train', use_correlated=False, augment=False)
    val_dataset = Pix2PixDataset('data/val', use_correlated=False, augment=False)
    test_dataset = Pix2PixDataset('data/test', use_correlated=False, augment=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              collate_fn=utils.custom_collate)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=True,
                            collate_fn=utils.custom_collate)
    
    test_loader = DataLoader(test_dataset,
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
