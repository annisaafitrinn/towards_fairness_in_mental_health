import torch.nn as nn
import torch

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, lstm_units, output_units, dropout_rate=0.5):
        super(CNN_LSTM, self).__init__()

        # Convolutional layer + BatchNorm + ReLU + Pooling
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_units, batch_first=True)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_units, output_units)

    def forward(self, x):
        # x: (batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Transpose for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM output: hn has shape (num_layers * num_directions, batch, hidden_size)
        x, (hn, cn) = self.lstm(x)

        # Apply dropout to the last hidden state
        x = self.dropout(hn[-1])

        # Final classification
        x = self.fc(x)

        return x
