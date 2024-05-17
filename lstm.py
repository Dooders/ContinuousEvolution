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

    def forward(self, x):
        # Forward pass through LSTM
        output, _ = self.lstm(x)
        # Pass the output of the last timestep through the linear layer
        return self.linear(output[:, -1])
