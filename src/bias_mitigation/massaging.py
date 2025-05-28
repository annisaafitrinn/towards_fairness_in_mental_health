import numpy as np
from utils import *
from torch.utils.data import DataLoader
from models import *

def main():
    base_path = 'data'

    # Load original labels and gender info
    train_labels = np.load(f"{base_path}/train/labels.npy")
    train_gender = np.load(f"{base_path}/train/gender.npy")

    # Perform label massaging
    massage_labels(train_labels, train_gender, base_path)

    # Load features to compute normalization params
    train_features = np.load(f"{base_path}/train/features.npy")
    train_mean = train_features.mean()
    train_std = train_features.std() + 1e-8

    # Load massaged train dataset
    train_dataset = load_data(base_path, 'train', train_mean, train_std)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = load_data1("val", train_mean, train_std)
    test_dataset = load_data1("test", train_mean, train_std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set seed for reproducibility
    set_seed(42)

    test_gender = np.load(base_path + "/test/gender.npy")
     
      # 1. Simple CNN
    set_seed(42)
    model_cnn_mas, metrics_cnn_mas = train_model(
        model_class=SimpleCNN,
        model_name="cnn",
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )
    set_seed(42)
    evaluate_model(
        model=model_cnn_mas,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )

    # 2. CNN + LSTM
    set_seed(42)
    model_cnn_lstm_mas, metrics_cnn_lstm_mas = train_model(
        model_class=CNN_LSTM,
        model_name="cnn_lstm",
        lstm_units=64,
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )
    set_seed(42)
    evaluate_model(
        model=model_cnn_lstm_mas,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )

    # 3. CNN + GRU + Attention
    set_seed(42)
    model_cnn_gru_mas, metrics_cnn_gru_mas = train_model(
        model_class=CNN_GRU_Attn,
        model_name="cnn_gru_attention",
        gru_units=64,
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )
    set_seed(42)
    evaluate_model(
        model=model_cnn_gru_mas,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )

if __name__ == "__main__":
    main()