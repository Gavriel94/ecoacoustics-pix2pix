"""
Script where model hyperparameters and classes are instantiated for training
of the conditional Generative Adverserial Network.

Creates a `runs` folder in the dataset and populates it with samples of
training and validation data during training. Each new run generates a new directory
so images are organised and easily viewable.

A `model` folder is added that saves the generator and discriminator parameters
so the model can be evaluated.
"""

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from pix2pix.dataset_train import Pix2PixTrainDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from pix2pix.custom_loss import Pix2PixLoss
from pix2pix.train_cGAN import train_cGAN
import pix2pix.utilities as utils
import config
import os


def main():
    train_dataset = Pix2PixTrainDataset(dataset=os.path.join(config.DATASET_ROOT, 'train'),
                                        use_correlated=False)
    val_dataset = Pix2PixTrainDataset(dataset=os.path.join(config.DATASET_ROOT, 'val'),
                                      use_correlated=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS,
                              shuffle=True,
                              collate_fn=utils.train_collate)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS,
                            shuffle=True,
                            collate_fn=utils.train_collate)

    disc = Discriminator(in_ch=1).to(config.DEVICE)
    gen = Generator(in_ch=1, features=64).to(config.DEVICE)
    # setting betas reduce momentum (used in the paper)
    optim_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = Pix2PixLoss(alpha=0.8)

    train_cGAN(disc, gen, train_loader, val_loader,
               optim_disc, optim_gen, l1_loss, config.L1_LAMBDA,
               bce, config.NUM_EPOCHS, config.DEVICE, save_dir=config.DATASET_ROOT,
               accumulation_steps=config.ACCUMULATION_STEPS)


if __name__ == '__main__':
    main()
