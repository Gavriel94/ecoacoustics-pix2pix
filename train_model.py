"""
Script where model hyperparameters and classes are instantiated for training
of the conditional Generative Adverserial Network.

Creates a `train_runs` folder in the dataset and populates it with samples of
training and validation data during training. Each new run generates a new directory
so images are organised and easily viewable.

A `model` folder is added that saves the generator and discriminator parameters
so the model can be evaluated.
"""

import os

from torch import nn, optim
from torch.utils.data import DataLoader

import config
import pix2pix.model_utils as m_utils
from pix2pix.dataset_train import Pix2PixTrainingDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator_regular_conv import Generator as GenRegConv
from pix2pix.generator_transpose_conv import Generator as GenRegTranspose
from pix2pix.train import train_cGAN


def main():
    train_dataset = Pix2PixTrainingDataset(dataset=os.path.join(config.DATASET_ROOT, 'train'),
                                           use_correlated=False)
    val_dataset = Pix2PixTrainingDataset(dataset=os.path.join(config.DATASET_ROOT, 'val'),
                                         use_correlated=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS,
                              shuffle=True,
                              collate_fn=m_utils.train_collate)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS,
                            shuffle=True,
                            collate_fn=m_utils.train_collate)

    disc = Discriminator(in_ch=1).to(config.DEVICE)
    gen = GenRegConv(in_ch=1, features=64).to(config.DEVICE)
    # setting betas to reduce momentum
    optim_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    train_cGAN(disc, gen, train_loader, val_loader,
               optim_disc, optim_gen, config.CUSTOM_LOSS, config.L1_LAMBDA,
               bce, config.NUM_EPOCHS, config.DEVICE, save_dir=config.DATASET_ROOT,
               accumulation_steps=config.ACCUMULATION_STEPS)


if __name__ == '__main__':
    main()
