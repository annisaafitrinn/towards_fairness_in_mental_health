import numpy as np
from utils import *
from models import *

def main():
    base_path = 'data'

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

    model_cnn_reg, metrics_cnn_reg = train_model(
        model_class=SimpleCNN,
        model_name="cnn",
        dropout_rate=0.5,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )

 
    evaluate_model(
        model=model_cnn_reg,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )


    model_cnn_lstm_reg, metrics_model_cnn_lstm_reg = train_model(
        model_class=CNN_LSTM,
        model_name="cnn_lstm",
        lstm_units=64,
        dropout_rate=0.4,
        num_epochs=100,

)

    evaluate_model(
    model=model_cnn_lstm_reg,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender",  # or any attribute like "Age", "Race"
    target_names=["Class 0", "Class 1"]
)
    

    model_cnn_gru_reg, metrics_cnn_gru_reg = train_model(
    model_class=CNN_GRU_Attn,
    model_name="cnn_gru_attention",
    gru_units=64,
    dropout_rate=0.4,
    num_epochs=100,
)
    
    evaluate_model(
    model=model_cnn_gru_reg,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender", 
    target_names=["Class 0", "Class 1"]
)

if __name__ == "__main__":
    main()