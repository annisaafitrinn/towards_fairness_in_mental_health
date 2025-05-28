import numpy as np
from utils import *
from models import *


def main():
    base_path = 'data'

    train_features, train_labels, train_gender = load_features_labels(base_path, 'train')

    aif_dataset = create_aif360_dataset(train_labels, train_gender)
    beta_list = get_instance_weights(aif_dataset)
    train_dataset = load_data1(base_path, 'train')  

    train_dataset_indexed = IndexedDataset(train_dataset)
    train_loader = get_dataloader(train_dataset_indexed, batch_size=32, shuffle=True)
    
    train_mean = train_features.mean()
    train_std = train_features.std() + 1e-8

    val_dataset = load_data1("val", train_mean, train_std)
    test_dataset = load_data1("test", train_mean, train_std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    set_seed(42)
    test_gender = np.load(base_path + "/test/gender.npy")

    model_cnn_rew, metrics_cnn_rew = train_model(
        model_class=SimpleCNN,
        model_name="cnn",
        dropout_rate=0.5,
        num_epochs=100,
        reweighting=True,
        beta_list=beta_list,
        learning_rate=0.001,
        weight_decay=0.001,
        patience=10,
        train_loader=train_loader
    )

    test_gender = np.load(f"{base_path}/test/gender.npy")

    evaluate_model(
        model=model_cnn_rew,
        test_loader=test_loader,
        test_sensitive_attr=test_gender,
        group_name="Gender",
        target_names=["Class 0", "Class 1"]
    )


    model_cnn_lstm_rew, metrics_model_cnn_lstm_rew = train_model(
        model_class=CNN_LSTM,
        model_name="cnn_lstm",
        lstm_units=64,
        dropout_rate=0.4,
        num_epochs=100,
        reweighting=True,
        beta_list=beta_list
)

    evaluate_model(
    model=model_cnn_lstm_rew,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender",  # or any attribute like "Age", "Race"
    target_names=["Class 0", "Class 1"]
)
    

    model_cnn_gru_rew, metrics_cnn_gru_rew = train_model(
    model_class=CNN_GRU_Attn,
    model_name="cnn_gru_attention",
    gru_units=64,
    dropout_rate=0.4,
    num_epochs=100,
    reweighting=True,
    beta_list=beta_list
)
    
    evaluate_model(
    model=model_cnn_gru_rew,
    test_loader=test_loader,
    test_sensitive_attr=test_gender,
    group_name="Gender", 
    target_names=["Class 0", "Class 1"]
)
    
if __name__ == "__main__":
    main()
