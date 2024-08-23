import os

import config
import create_dataset
import test_model
import train_model


def main():
    if not os.path.exists(config.DATASET_ROOT):
        create_dataset.main()

    train_model.main()
    test_model.main()


if __name__ == '__main__':
    main()
