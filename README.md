# Intelligent Acoustic Monitoring: AI-Based Analysis of Microphone Variability in Natural Environments

This project applies artificial intelligence to analyse and infer differences between [high-end](https://www.wildlifeacoustics.com/products/song-meter-sm4) and [low-end](https://www.wildlifeacoustics.com/products/song-meter-micro) SM microphones used in a rewilding project.

By leveraging machine learning techniques, this study aims to identify differences in species detection and ambient noise between these microphones when deployed in a forest setting. This research provides insight into the suitability of different types of microphones for ecological monitoring and the potential impact on data collection in environmental studies, potentially broadening possibilities for high quality species monitoring with reduced expense.

## Use

### DATA_ROOT

The path to the directory containing all of the raw data
The data must be organised in this way

```python
raw_data/
└── year/
    ├── microphone_1/
    │   ├── location_1/
    │   └── location_2/
    └── microphone_2/
        ├── location_1/
        └── location_2/
```

### DATASET_ROOT

The dataset will be saved with this structure in the folder where you set `DATASET_ROOT` equal to a directory.

```python
data/
├── year/
│   ├── location_1/
│   └── location_2/
├── analysis/
│   ├── microphone_1_data.csv
│   └── microphone_2_data.csv
├── dataset/
└── spectrograms/
    └── year/
        ├── location_1/
        └── location_2/
```

## Notes

Generating small spectrograms make longer spectrograms squash and may increase inaccuracy. Would recommend not to use `correlated` if so.