import os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from pix2pix.dataset import SpectrogramDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from dataset import create_dataset
from pix2pix.train import train_model
from pix2pix import config as cfg
import pix2pix.utilities as utils
from pix2pix.ds import Pix2PixDataset


def main():
    # utils.spectrogram_to_audio('data/spectrograms/2023_11/PLI3/PLI3_20231129_142200.png', 'data/spectrograms/2023_11/PLI3/params/PLI3_20231129_142200.json', 'tmp/e.wav')
    create_new_dataset = True

    if create_new_dataset:
        os.makedirs(cfg.DATASET_PATH, exist_ok=True)
        print('Creating a new dataset in \'data/\'\n')
        create_dataset(data_root=cfg.DATA_ROOT,
                       dataset_root=cfg.DATASET_ROOT,
                       verbose=True)

    files = utils.get_files(cfg.DATASET_PATH, include_correlated=False)[:10]
    
    
        
    # # get files from dataset
    # files = utils.get_files(cfg.DATASET_PATH, include_correlated=False)

    # # * used for debugging - alter the size of dataset
    # files = files[:10]

    # train, val, test = utils.split_data(files, 0.8)

    # train_dataset = Pix2PixDataset(train, augment=False)
    # val_dataset = Pix2PixDataset(val, augment=False)
    # test_dataset = Pix2PixDataset(test, augment=False)
    
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.BATCH_SIZE,
    #                           num_workers=cfg.NUM_WORKERS,
    #                           shuffle=True,
    #                           collate_fn=utils.custom_collate)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=cfg.BATCH_SIZE,
    #                         num_workers=cfg.NUM_WORKERS,
    #                         shuffle=False,
    #                         collate_fn=utils.custom_collate)
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=cfg.BATCH_SIZE,
    #                          num_workers=cfg.NUM_WORKERS,
    #                          shuffle=False,
    #                          collate_fn=utils.custom_collate)

    # disc = Discriminator(in_ch=1).to(cfg.DEVICE)
    # gen = Generator(in_ch=1, features=64).to(cfg.DEVICE)
    # # setting betas reduce momentum (used in the paper)
    # optim_disc = optim.Adam(disc.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    # optim_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    # BCE = nn.BCEWithLogitsLoss()
    # L1_LOSS = nn.L1Loss()
    # disc_loss, gen_loss, l1_loss = train_model(disc, gen, train_loader,
    #                                            optim_disc, optim_gen, L1_LOSS,
    #                                            BCE)
    # utils.plot_loss(disc_loss, gen_loss, l1_loss)


if __name__ == '__main__':
    main()
