# Fairness in Depression Detection
This repository contains code and methods for training and evaluating models on EEG for depression detection task, with a focus on fairness and bias mitigation. The project supports several deep learning architectures and bias mitigation techniques.

## Get Started

Clone this repository to your local machine using:

```bash
git clone https://github.com/annisaafitrinn/fairness_in_depression_detection.git
cd your-repo-name
```

## Directory Structure
```bash
├── README.md
├── data
│   └── README.md                # Data description and access info (data not public)
├── requirements.txt             # Project dependencies
└── src                          # Source code folder
    ├── baselines.py             # Baseline model training and evaluation scripts
    ├── bias_mitigation          # Bias mitigation methods and algorithms
    │   ├── data_augmentation.py
    │   ├── eop.py               # Equalized Odds Postprocessing
    │   ├── massaging.py
    │   ├── regularization.py    # Fairness-aware loss functions
    │   ├── reweighing.py
    │   └── roc.py               # Reject Option Classification
    ├── models                   # Deep learning model definitions
    │   ├── __init__.py
    │   ├── cnn_gru.py
    │   ├── cnn_lstm.py
    │   └── cnn_model.py
    ├── preprocessing            # EEG signal preprocessing utilities
    │   └── signal_preprocessing.py
    └── utils.py                 # Utility functions for data loading, metrics, etc.
```

## Requirements
Install the dependencies using:
```bash
pip install -r requirements.txt
```
## Data
The EEG dataset used in this project is not publicly available due to privacy restrictions. See data/README.md for more details.

## Usage
Each bias mitigation method has its own script in src/bias_mitigation/. For example, to train and evaluate reweighing method:
```bash
python src/bias_mitigation/reweighing.py
```

To run data augmentation based mitigation:

```bash
python src/bias_mitigation/data_augmentation.py
```

Modify or extend these scripts to customize trainin or mitigation strategies.

