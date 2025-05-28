import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_GRU_Attn(nn.Module):
    def __init__(self, input_channels, gru_units, output_units, dropout_rate=0.5):
        super(CNN_GRU_Attn, self).__init__()

        # First 1D Convolution + Pooling
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second 1D Convolution + Pooling
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # GRU
        self.gru = nn.GRU(input_size=128, hidden_size=gru_units, batch_first=True, bidirectional=True)

        # Attention layer
        self.attn = nn.Linear(gru_units * 2, 1)

        # Dropout and final classification
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(gru_units * 2, output_units)

    def forward(self, x):
        # x: (batch_size, channels, sequence_length)
        x = self.pool1(F.relu(self.conv1(x)))   # → (batch_size, 64, seq_len/2)
        x = self.pool2(F.relu(self.conv2(x)))   # → (batch_size, 128, seq_len/4)

        # Prepare for GRU: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # GRU output
        gru_out, _ = self.gru(x)  # gru_out: (batch, seq_len, gru_units*2)

        # Attention weights
        attn_weights = F.softmax(self.attn(gru_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)   # (batch, gru_units*2)

        # Dropout + FC
        out = self.dropout(context)
        out = self.fc(out)
        return out
