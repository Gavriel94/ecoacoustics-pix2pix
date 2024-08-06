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

from PIL import Image


def main():
    if not os.path.isdir(cfg.DATASET_PATH) or len(os.listdir(cfg.DATASET_PATH)) == 0:
        print('Creating a new dataset in \'data/\'')
        create_dataset(data_root='raw_data/',
                       dataset_root='data/',
                       analysis=False,
                       matched_summaries=cfg.MATCHED_SUMMARIES,
                       copied_recordings=cfg.COPIED_RECORDINGS,
                       spectrogram_paths=cfg.SPECTROGRAM_PATHS,
                       verbose=True)

    # initialise DataLoaders
    files = utils.get_files(cfg.DATASET_PATH, include_correlated=False)
    # print(files)
    
    
    # * used for debugging - alter the size of dataset
    files = files[:2]

    for file_path in files:
        file = Image.open(file_path)
        save_path = file_path.split('/')[2]  # includes .png
        file.save(f'tmp/{save_path}')
    
    # train, val, test = utils.split_data(files, 0.8)

    # train_dataset = SpectrogramDataset(train, augment=False, canvas_size=4096)
    # # print(train_dataset.__getitem__(0))
    # val_dataset = SpectrogramDataset(val, augment=False, canvas_size=4096)
    # test_dataset = SpectrogramDataset(test, augment=False, canvas_size=4096)
    # # print(type(train_dataset))
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.BATCH_SIZE,
    #                           num_workers=cfg.NUM_WORKERS,
    #                           shuffle=True)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=cfg.BATCH_SIZE,
    #                         num_workers=cfg.NUM_WORKERS,
    #                         shuffle=False)
    # test_loader = DataLoader(test_dataset,
    #                          batch_size=cfg.BATCH_SIZE,
    #                          num_workers=cfg.NUM_WORKERS,
    #                          shuffle=False)

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
