import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io
from scipy import signal

from aif360.algorithms.postprocessing import (
    RejectOptionClassification,
    EqOddsPostprocessing,
)
from aif360.algorithms.preprocessing import Reweighing


def evaluate_model(model, test_loader, test_sensitive_attr, 
                   group_name="Gender", target_names=['Class 0', 'Class 1'], 
                   apply_post_roc=False, apply_post_eqodds = False):
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_sensitive_attr = np.array(test_sensitive_attr)

    df = pd.DataFrame({
        'label': all_labels,
        'score': all_probs,
        'predicted': all_preds,
        group_name.lower(): test_sensitive_attr
    })

    dataset_true = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df.copy(),
        label_names=['label'],
        protected_attribute_names=[group_name.lower()]
    )

    dataset_pred = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df.copy(),
        label_names=['label'],
        protected_attribute_names=[group_name.lower()]
    )
    dataset_pred.labels = df['predicted'].values.reshape(-1, 1)
    dataset_pred.scores = df['score'].values.reshape(-1, 1)

    if apply_post_roc:
        privileged_groups = [{group_name.lower(): 0}]
        unprivileged_groups = [{group_name.lower(): 1}]

        roc = RejectOptionClassification(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            low_class_thresh=0.4,
            high_class_thresh=0.7,
            num_class_thresh=100,
            num_ROC_margin=50,
            metric_name="Statistical parity difference",  # or another fairness metric
            metric_ub=0.05,
            metric_lb=-0.05
        )

        roc = roc.fit(dataset_true, dataset_pred)
        dataset_pred = roc.predict(dataset_pred)
        
    if apply_post_eqodds:
        privileged_groups = [{group_name.lower(): 0}]
        unprivileged_groups = [{group_name.lower(): 1}]
        eq_odds = EqOddsPostprocessing(
                privileged_groups=privileged_groups,
                unprivileged_groups=unprivileged_groups,
                seed=42
            )
    
        eq_odds = eq_odds.fit(dataset_true, dataset_pred)
        dataset_pred = eq_odds.predict(dataset_pred)
        
    final_preds = dataset_pred.labels.ravel().astype(int)

    print("Classification Report:")
    print(classification_report(all_labels, final_preds, target_names=target_names))

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=[{group_name.lower(): 0}],
        privileged_groups=[{group_name.lower(): 1}]
    )

    print("\n Fairness Metrics (AIF360):")
    print(f"  ➤ Disparate Impact: {metric.disparate_impact():.3f}")
    print(f"  ➤ Statistical Parity Difference: {metric.statistical_parity_difference():.3f}")
    print(f"  ➤ Equal Opportunity Difference (TPR): {metric.equal_opportunity_difference():.3f}")
    print(f"  ➤ False Positive Rate Difference (FPR): {metric.false_positive_rate_difference():.3f}")
    print(f"  ➤ Average Odds Difference: {metric.average_odds_difference():.3f}")

    acc_priv = metric.accuracy(privileged=True)
    acc_unpriv = metric.accuracy(privileged=False)
    print(f"  ➤ Equalized Accuracy Difference: {acc_unpriv - acc_priv:.3f}")



def train_model(model_class, model_name, num_epochs=70, batch_size=32,
                learning_rate=0.0001, weight_decay=0.00001,
                reweighting=False, beta_list=None,
                patience=10,
                **model_kwargs):
    """
    Training function with support for instance reweighting, early stopping, and learning rate scheduling.

    Args:
        model_class: The model class to instantiate.
        model_name (str): Identifier for model type (e.g., 'cnn').
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        reweighting (bool): If True, use instance weighting for fairness.
        beta_list (np.array or torch.Tensor): Weights for each sample.
        patience (int): Number of epochs to wait for improvement before stopping.
        model_kwargs: Additional arguments to pass to the model constructor.

    Returns:
        model: Trained model.
        (train_losses, val_losses, train_accuracies, val_accuracies): Training history.
    """
    input_channels = 16
    output_units = len(np.unique(train_dataset.tensors[1].numpy()))
    seq_len = train_data.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model construction
    if model_name == "cnn":
        model = model_class(
            input_channels=input_channels,
            output_units=output_units,
            input_seq_len=seq_len,
            **model_kwargs
        )
    elif model_name == "lstm":
        model = model_class(
            input_channels=input_channels,
            output_units=output_units,
            input_seq_len=seq_len,
            **model_kwargs  # you can pass hidden_size, num_layers, etc.
        )
    else:
        model = model_class(
            input_channels=input_channels,
            output_units=output_units,
            **model_kwargs
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    criterion = nn.CrossEntropyLoss(reduction='none') if reweighting else nn.CrossEntropyLoss()

    if reweighting:
        assert beta_list is not None, "Beta list must be provided if reweighting is True"
        sample_weights = torch.tensor(beta_list, dtype=torch.float32)
    else:
        sample_weights = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)

            if reweighting:
                indices = batch_idx * batch_size + torch.arange(X_batch.size(0))
                batch_weights = sample_weights[indices].to(device)
                losses = criterion(outputs, y_batch)
                loss = (batch_weights * losses).mean()
            else:
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()

        train_acc = 100 * correct / len(train_dataset)
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                val_loss += nn.CrossEntropyLoss()(val_outputs, y_val).item()
                val_correct += (val_outputs.argmax(dim=1) == y_val).sum().item()

        val_acc = 100 * val_correct / len(val_dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Restore best model
    model.load_state_dict(best_model_state)
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

def load_data1(split, mean=None, std=None, standardize=True):
    features = np.load(f"{base_path}/{split}/features.npy")
    labels = np.load(f"{base_path}/{split}/labels.npy")

    if standardize and mean is not None and std is not None:
        features = (features - mean) / std

    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def load_data(mean=None, std=None, standardize=True):
    features = train_features_balanced
    labels = train_labels_balanced

    if standardize and mean is not None and std is not None:
        features = (features - mean) / std

    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mixup_data(x_i, x_j, y_i, y_j, alpha=0.2):
    '''Generates new samples using Mixup.'''
    lam = np.random.beta(alpha, alpha)
    x_new = lam * x_i + (1 - lam) * x_j
    y_new = lam * y_i + (1 - lam) * y_j
    return x_new, y_new

def augment_minority_class(features, labels, num_samples_needed):
    augmented_features = []
    augmented_labels = []
    for _ in range(num_samples_needed):
        i, j = np.random.choice(len(features), 2, replace=False)
        x_i, y_i = features[i], labels[i]
        x_j, y_j = features[j], labels[j]
        x_new, y_new = mixup_data(x_i, x_j, y_i, y_j)
        augmented_features.append(x_new)
        augmented_labels.append(y_new)
    return np.array(augmented_features), np.array(augmented_labels)

def load_data_mas(base_path, split, mean=None, std=None, standardize=True):
    '''
    Load features and (possibly massaged) labels from numpy files, 
    return a Torch TensorDataset.
    '''
    features = np.load(f"{base_path}/{split}/features.npy")
    labels = np.load(f"{base_path}/{split}/labels_massaged.npy")  # Note: use massaged labels
    
    if standardize and mean is not None and std is not None:
        features = (features - mean) / std

    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def massage_labels(train_labels, train_gender, base_path):
    '''
    Perform label massaging by flipping labels between favored and deprived groups to reduce bias.
    Saves the massaged labels to disk.
    '''
    male_mask = train_gender == 0
    female_mask = train_gender == 1

    p_male = train_labels[male_mask].mean()  
    p_female = train_labels[female_mask].mean() 

    favored_group = 0 if p_male > p_female else 1
    deprived_group = 1 - favored_group

    print(f"Favored: {'Male' if favored_group == 0 else 'Female'}, Deprived: {'Male' if deprived_group == 0 else 'Female'}")

    gap = abs(p_male - p_female)
    total_to_flip = int(gap * min(np.sum(male_mask), np.sum(female_mask)))
    print(f"Relabeling {total_to_flip} samples")

    favored_idxs = np.where((train_gender == favored_group) & (train_labels == 1))[0]
    favored_to_flip = np.random.choice(favored_idxs, total_to_flip, replace=False)
    train_labels[favored_to_flip] = 0

    deprived_idxs = np.where((train_gender == deprived_group) & (train_labels == 0))[0]
    deprived_to_flip = np.random.choice(deprived_idxs, total_to_flip, replace=False)
    train_labels[deprived_to_flip] = 1

    np.save(f"{base_path}/train/labels_massaged.npy", train_labels)

def load_features_labels(base_path, split='train', massaged=False):
    features = np.load(f"{base_path}/{split}/features.npy")
    if massaged:
        labels = np.load(f"{base_path}/{split}/labels_massaged.npy")
    else:
        labels = np.load(f"{base_path}/{split}/labels.npy")
    gender = np.load(f"{base_path}/{split}/gender.npy")
    return features, labels, gender

def create_aif360_dataset(labels, protected_attr, favorable_label=0, unfavorable_label=1):
    df = pd.DataFrame({
        'label': labels,
        'gender': protected_attr
    })
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['gender'],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label
    )
    return dataset

def get_instance_weights(dataset, privileged_groups=[{'gender': 0}], unprivileged_groups=[{'gender': 1}]):
    RW = Reweighing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)
    transformed_dataset = RW.fit_transform(dataset)
    return transformed_dataset.instance_weights

class IndexedDataset(Dataset):
    """Dataset wrapper that returns (x, y, idx) for accessing instance weights"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, idx
    
def get_dataloader(dataset, batch_size=32, shuffle=True):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model_reg(model, test_loader, test_sensitive_attr, group_name="Gender", target_names=['Class 0', 'Class 1'], apply_post_roc=False, tau=0.6):
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels,y in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class 1 (Depressed)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_sensitive_attr = np.array(test_sensitive_attr)

    if apply_post_roc:
        def apply_roc(probabilities, genders, tau=0.6, unprivileged_value=0, favorable_label=0):
            original_preds = (probabilities >= 0.5).astype(int)
            uncertain = (probabilities >= (1 - tau)) & (probabilities <= tau)
            flip_mask = uncertain & (genders == unprivileged_value)
            original_preds[flip_mask] = favorable_label
            return original_preds

        all_preds = apply_roc(all_probs, test_sensitive_attr, tau=tau)

    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    df = pd.DataFrame({
        'label': all_labels,
        'predicted': all_preds,
        group_name.lower(): test_sensitive_attr
    })

    dataset = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df,
        label_names=['label'],
        protected_attribute_names=[group_name.lower()]
    )

    predicted_dataset = dataset.copy()
    predicted_dataset.labels = df['predicted'].values.reshape(-1, 1)

    privileged_groups = [{group_name.lower(): 1}]
    unprivileged_groups = [{group_name.lower(): 0}]

    metric = ClassificationMetric(
        dataset,
        predicted_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    print("\n📈 Fairness Metrics (AIF360):")
    print(f"  ➤ Disparate Impact: {metric.disparate_impact():.3f}")
    print(f"  ➤ Statistical Parity Difference: {metric.statistical_parity_difference():.3f}")
    print(f"  ➤ Equal Opportunity Difference (TPR): {metric.equal_opportunity_difference():.3f}")
    print(f"  ➤ False Positive Rate Difference (FPR): {metric.false_positive_rate_difference():.3f}")
    print(f"  ➤ Average Odds Difference: {metric.average_odds_difference():.3f}")

    acc_priv = metric.accuracy(privileged=True)
    acc_unpriv = metric.accuracy(privileged=False)
    print(f"  ➤ Equalized Accuracy Difference: {acc_unpriv - acc_priv:.3f}")


def train_model_reg(model_class, model_name, train_dataset, val_dataset,
                    num_epochs=100, batch_size=32,
                    learning_rate=1e-4, weight_decay=1e-5,
                    lambda_eopp=0.5, lambda_eodd=0.5,
                    patience=10,
                    fairness_regularized_loss=None,
                    **model_kwargs):
    """
    Training function with fairness regularization.

    Args:
        model_class: Model class to instantiate.
        model_name (str): Model type identifier.
        train_dataset: Training dataset, must return (X, y, g).
        val_dataset: Validation dataset, must return (X, y, g).
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Optimizer LR.
        weight_decay (float): Optimizer weight decay.
        lambda_eopp (float): Weight for Equal Opportunity regularizer.
        lambda_eodd (float): Weight for Equal Odds regularizer.
        patience (int): Early stopping patience.
        fairness_regularized_loss (callable): Function(outputs, y, g, λ1, λ2) → loss.
        model_kwargs: Additional arguments for model constructor.

    Returns:
        Trained model and history tuple.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_labels = np.unique(val_dataset[:][1].numpy())
    output_units = len(unique_labels)
    model = model_class(output_units=output_units, **model_kwargs)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_acc = 0
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            # Expecting train_dataset to return (X, y, g)
            X_batch, y_batch, g_batch = batch
            X_batch, y_batch, g_batch = X_batch.to(device), y_batch.to(device), g_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = fairness_regularized_loss(outputs, y_batch, g_batch, lambda_eopp, lambda_eodd)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total_samples += X_batch.size(0)

        train_loss = total_loss / total_samples
        train_acc = 100 * correct / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                X_val, y_val, g_val = batch
                X_val, y_val, g_val = X_val.to(device), y_val.to(device), g_val.to(device)

                val_outputs = model(X_val)
                loss_val = fairness_regularized_loss(val_outputs, y_val, g_val, lambda_eopp, lambda_eodd)
                val_loss += loss_val.item() * X_val.size(0)

                preds_val = val_outputs.argmax(dim=1)
                val_correct += (preds_val == y_val).sum().item()
                val_samples += X_val.size(0)

        val_loss /= val_samples
        val_acc = 100 * val_correct / val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    model.load_state_dict(best_model_state)
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

def process_and_segment_eeg(eeg_path, save_path, segment_length=8):
    """
    Processes EEG .mat files by extracting selected channels, applying filtering,
    re-referencing, segmenting into 8-second windows, and saving as .npy files.

    Removes 50 Hz power line noise using a notch filter.

    Parameters:
        eeg_path (str): Path to the directory containing .mat files.
        save_path (str): Path to save processed .npy files.
        segment_length (int): Segment duration in seconds (default: 8s).
    """
    os.makedirs(save_path, exist_ok=True)

    mat_files = [f for f in os.listdir(eeg_path) if f.endswith('.mat')]

    for mat_file in mat_files:
        file_path = os.path.join(eeg_path, mat_file)
        data = scipy.io.loadmat(file_path)

        eeg_key = [key for key in data.keys() if 'mat' in key][0]
        eeg_data = data[eeg_key]  # Shape: (129, timepoints)
        sampling_rate = int(data['samplingRate'][0][0])

        # Select only the required 16 channels
        eeg_selected = eeg_data[channels_to_extract, :]

        lowcut, highcut = 0.5, 50.0
        nyquist = 0.5 * sampling_rate
        b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        filtered_channels = np.array([signal.filtfilt(b, a, eeg_selected[i, :]) for i in range(eeg_selected.shape[0])])

        # Notch filter at 50 Hz to remove power line noise
        notch_freq = 50.0  # Power line frequency
        Q = 30  # Quality factor
        b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=sampling_rate)
        notch_filtered_channels = np.array([signal.filtfilt(b_notch, a_notch, ch) for ch in filtered_channels])

        # Average reference
        avg_reference = np.mean(notch_filtered_channels, axis=0)
        referred_channels = notch_filtered_channels - avg_reference

        # Segmenting into 8-second windows
        segment_samples = segment_length * sampling_rate
        num_segments = referred_channels.shape[1] // segment_samples

        segmented_data = np.array([referred_channels[:, i * segment_samples: (i + 1) * segment_samples]
                                   for i in range(num_segments)])

        save_filename = os.path.join(save_path, f"{mat_file.replace('.mat', '.npy')}")
        np.save(save_filename, segmented_data)

def process_and_segment(eeg_path, save_path):
    process_and_segment_eeg(eeg_path, save_path) 

def merge_segmented_data(segmented_eeg_path, merged_save_path, labels_save_path):
    segmented_files = [f for f in os.listdir(segmented_eeg_path) if f.endswith('.npy')]

    all_subject_data = []
    all_subject_labels = []

    for file in segmented_files:
        file_path = os.path.join(segmented_eeg_path, file)
        eeg_data = np.load(file_path)
        subject_id = file.split('rest')[0][:8]

        all_subject_data.append(eeg_data)
        num_segments = eeg_data.shape[0]
        subject_labels = np.array([subject_id] * num_segments)
        all_subject_labels.append(subject_labels)

    merged_data = np.concatenate(all_subject_data, axis=0)
    merged_labels = np.concatenate(all_subject_labels, axis=0)

    print(f"Merged data shape: {merged_data.shape}")
    print(f"Merged labels shape: {merged_labels.shape}")

    np.save(merged_save_path, merged_data)
    np.save(labels_save_path, merged_labels)

    print(f"Merged EEG data saved to: {merged_save_path}")
    print(f"Merged labels saved to: {labels_save_path}")