import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset


class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_units, input_seq_len, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(dropout_rate)

        # Compute final feature size after conv + pooling
        dummy_input = torch.zeros(1, input_channels, input_seq_len)
        x = self.pool1(self.bn1(self.conv1(dummy_input)))
        x = self.pool2(self.bn2(self.conv2(x)))
        flattened_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(flattened_size, output_units)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
