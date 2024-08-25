import os

import config
import create_dataset
import train_model
import evaluate_model


def main():
    if not os.path.exists(config.DATASET_ROOT):
        create_dataset.main()

    train_model.main()
    evaluate_model.main()


if __name__ == '__main__':
    main()
