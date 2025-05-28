import numpy as np
from utils import *
from torch.utils.data import DataLoader
import torch
from models import *

def main():
    base_path = 'data'
    train_features = np.load(f'{base_path}/train/features.npy')
    train_labels = np.load(f'{base_path}/train/labels.npy')
    train_gender = np.load(f'{base_path}/train/gender.npy')

    # Define groups
    male_healthy = (train_labels == 0) & (train_gender == 0)
    male_depressed = (train_labels == 1) & (train_gender == 0)
    female_healthy = (train_labels == 0) & (train_gender == 1)
    female_depressed = (train_labels == 1) & (train_gender == 1)

    # Extract subgroup data
    mh_feat, mh_lab = train_features[male_healthy], train_labels[male_healthy]
    md_feat, md_lab = train_features[male_depressed], train_labels[male_depressed]
    fh_feat, fh_lab = train_features[female_healthy], train_labels[female_healthy]
    fd_feat, fd_lab = train_features[female_depressed], train_labels[female_depressed]

    # Calculate samples to augment
    samples_fh = len(mh_feat) - len(fh_feat)
    samples_fd = len(mh_feat) - len(fd_feat)
    samples_md = len(mh_feat) - len(md_feat)

    # Augment minority groups
    aug_fh_feat, aug_fh_lab = augment_minority_class(fh_feat, fh_lab, samples_fh)
    aug_fd_feat, aug_fd_lab = augment_minority_class(fd_feat, fd_lab, samples_fd)
    aug_md_feat, aug_md_lab = augment_minority_class(md_feat, md_lab, samples_md)

    # Concatenate augmented data
    train_features_balanced = np.concatenate(
        (train_features, aug_fh_feat, aug_fd_feat, aug_md_feat), axis=0
    )
    train_labels_balanced = np.concatenate(
        (train_labels, aug_fh_lab, aug_fd_lab, aug_md_lab), axis=0
    )

    train_mean, train_std = train_features_balanced.mean(), train_features_balanced.std() 
    train_dataset = load_data(train_features_balanced, train_labels_balanced, train_mean, train_std)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = load_data1("val", train_mean, train_std)
    test_dataset = load_data1("test", train_mean, train_std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    set_seed(42)

    model_cnn_aug, metrics_cnn_aug = train_model(
        model_class=SimpleCNN,
        model_name="cnn",
        dropout_rate=0.5,
        num_epochs=100,
            learning_rate = 0.0001,
        weight_decay = 0.0001,
        patience = 10
    )

    evaluate_model(
        model=model_cnn_aug,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )


   
    model_cnn_lstm_aug, metrics_cnn_lstm_aug = train_model(
        model_class=CNN_LSTM,
        model_name="cnn_lstm",
        lstm_units=128,
        dropout_rate=0.3,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )

    test_gender = np.load(base_path + "/test/gender.npy")

    evaluate_model(
        model=model_cnn_lstm_aug,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )

    model_cnn_gru_aug, metrics_cnn_gru_aug = train_model(
        model_class=CNN_GRU_Attn,
        model_name="cnn_gru_attention",
        gru_units=256,
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate = 0.001,
        weight_decay = 0.001,
        patience = 10
    )

    evaluate_model(
        model=model_cnn_gru_aug,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )

if __name__ == "__main__":
    main()
