import numpy as np
from utils import *
from models import *


def main():
    base_path = 'data'
    test_gender = np.load(base_path + "/test/gender.npy")
    set_seed(42)
    base_path = 'fairness-in-mental-health'

    train_features, train_labels, train_gender = load_features_labels(base_path, 'train')
    train_mean = train_features.mean()
    train_std = train_features.std() + 1e-8

    train_dataset = load_data1("train", train_mean, train_std)
    val_dataset = load_data1("val", train_mean, train_std)
    test_dataset = load_data1("test", train_mean, train_std)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    set_seed(42)
    test_gender = np.load(base_path + "/test/gender.npy")

    model_cnn, metrics_cnn = train_model(
        model_class=SimpleCNN,
        model_name="cnn",
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate = 0.0001,
        weight_decay = 0.001,
        patience = 10
    )

    evaluate_model(
        model=model_cnn,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        apply_post_roc = True,
        target_names=["Class 0", "Class 1"]
    )

    
    model_cnn_lstm, metrics_cnn_lstm = train_model(
    model_class=CNN_LSTM,
    model_name="cnn_lstm",
    lstm_units=64,
    dropout_rate=0.4,
    num_epochs=100,
      learning_rate = 0.001,
    weight_decay = 0.001,
    patience = 10
)
    
    evaluate_model(
    model=model_cnn_lstm,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender",
    apply_post_roc = True,
    target_names=["Class 0", "Class 1"]
)
    
    model_cnn_gru, metrics_cnn_gru = train_model(
    model_class=CNN_GRU_Attn,
    model_name="cnn_gru_attention",
    gru_units=64,
    dropout_rate=0.5,
    num_epochs=100,
    learning_rate = 0.001,
    weight_decay = 0.001,
    patience = 10
)
    
    evaluate_model(
    model=model_cnn_gru,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender",
    apply_post_roc = True,
    target_names=["Class 0", "Class 1"]
)


if __name__ == "__main__":
    main()
