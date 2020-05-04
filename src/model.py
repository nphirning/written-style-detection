import torch
import torch.nn as nn

INPUT_SIZE = 1000
TOKEN_SIZE = 768
HIDDEN_SIZE = 10

class LSTMBasedModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim, batch_size):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(TOKEN_SIZE, self.hidden_dim)
        self.out_nn = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.latent_dim), 
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1)
        )

    def init_hidden(self):
        return (
            torch.zeros(1, self.batch_size, self.hidden_dim),
            torch.zeros(1, self.batch_size, self.hidden_dim)
        )

    def forward(self, x1, x2):
        # The inputs `x1`, `x2` are tensors of size 
        #   (INPUT_LEN, BATCH_SIZE, EMBEDDING_SIZE).
        
        # Each LSTM output tensor has shape (INPUT_LEN, BATCH_SIZE, HIDDEN_DIM).
        lstm_out1, _ = self.lstm(x1)
        lstm_out2, _ = self.lstm(x2)

        # Concatenate the final outputs of each LSTM.
        linear_in = torch.cat((
            lstm_out1[-1].view(self.batch_size, -1), 
            lstm_out2[-1].view(self.batch_size, -1)
        ))

        # Pass the result through a 2-layer NN.
        return self.out_nn(linear_in).view(-1)

model = LSTM(1000, )

