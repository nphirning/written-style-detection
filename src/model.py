import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader

INPUT_SIZE = 1000
TOKEN_SIZE = 768
HIDDEN_SIZE = 10
LATENT_DIM = 10
BATCH_SIZE = 5
VOCAB_SIZE = 10000
NUM_EPOCHS = 3
LEARNING_RATE = 0.1

class LSTMBasedModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim, batch_size):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embeddings = nn.Embedding(VOCAB_SIZE, TOKEN_SIZE)
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

    def forward(self, x):
        # The input `x` is a Python list with 2 elements, each a tensor of size
        #   (INPUT_LEN, BATCH_SIZE, EMBEDDING_SIZE).
        
        # Look up embeddings.
        x1 = self.embeddings(x[0])
        x2 = self.embeddings(x[1])

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

# Create model, loss, and optimizer objects.
model = LSTMBasedModel(HIDDEN_SIZE, LATENT_DIM, BATCH_SIZE)
loss_func = nn.HingeEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Get input datasets.
def split_x_values(x):
    return [
        DataLoader(x_train[:, 0, :, :], batch_size=BATCH_SIZE,
        DataLoader(x_train[:, 1, :, :], batch_size=BATCH_SIZE)
    ]
x_train, y_train, x_dev, y_dev, x_test, y_test = get_data()
train_loaders = split_x_values(x_train)
label_loader = DataLoader(y_train, batch_size=BATCH_SIZE)

for i in range(NUM_EPOCHS):
    sum_loss = 0.0

    for j, batch in enumerate(train_loaders[0]):

        # Create batch input
        first_examples = torch.Tensor(train_loaders[0][j])
        second_examples = torch.Tensor(train_loaders[1][j])
        model_input = [first_examples, second_examples]

        # Initialize model.
        model.hidden = model.init_hidden()

        # Run model and compute loss.
        predictions = model(model_input)
        loss = loss_func(predictions, label_loader[j])
        sum_loss += loss.item() 

        # Update weights via backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch loss: %s" % sum_loss)


