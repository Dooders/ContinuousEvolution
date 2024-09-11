import torch
import torch.nn as nn
import torch.nn.init as init


class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.init_weights()

    def init_weights(self):
        # Initialize weights for the LSTM and linear layers
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param)
            elif "weight_hh" in name:
                init.orthogonal_(param)
        init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor has correct shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing

        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Get the last output
        last_output = lstm_out[:, -1, :]

        # Forward pass through linear layer
        linear_out = self.linear(last_output)

        return linear_out
