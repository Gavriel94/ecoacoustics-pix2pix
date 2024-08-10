from torch import optim
from torch import nn
from torch.utils.data import DataLoader

from pix2pix.custom_dataset import Pix2PixDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from pix2pix.custom_loss import CustomL1Loss
from pix2pix.train import train
import pix2pix.utilities as utils

DATASET = 'data_test/'
DEVICE = utils.set_device('mps')
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 15
L1_LAMBDA = 100

# Gradients accumulate (accumulation_steps * batch_size) steps before updating weights
ACCUMULATION_STEPS = 4
# epoch where progress information prints
DISPLAY_EPOCH = 2


def main():
    train_dataset = Pix2PixDataset(dataset='data_test/train', use_correlated=False)
    val_dataset = Pix2PixDataset(dataset='data_test/val', use_correlated=False)
    # test_dataset = Pix2PixDataset(dataset='data_test/test', use_correlated=False)

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

    # test_loader = DataLoader(test_dataset,
    #                          batch_size=BATCH_SIZE,
    #                          num_workers=NUM_WORKERS,
    #                          shuffle=True,
    #                          collate_fn=utils.custom_collate)

    disc = Discriminator(in_ch=1).to(DEVICE)
    gen = Generator(in_ch=1, features=64).to(DEVICE)
    # setting betas reduce momentum (used in the paper)
    optim_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1_loss = CustomL1Loss()
    train(disc, gen, train_loader, val_loader, optim_disc, optim_gen, l1_loss, L1_LAMBDA,
          bce, NUM_EPOCHS, DEVICE, save_dir=DATASET, accumulation_steps=ACCUMULATION_STEPS, display_epoch=DISPLAY_EPOCH)


if __name__ == '__main__':
    main()
