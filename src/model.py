import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
import os
import numpy as np


INPUT_SIZE = 1000
TOKEN_SIZE = 768
HIDDEN_SIZE = 10
LATENT_DIM = 10
BATCH_SIZE = 5
VOCAB_SIZE = 10000
NUM_EPOCHS = 3
LEARNING_RATE = 0.1




class BertBasedBiameseModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim, batch_size):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        self.out_nn = nn.Sequential(
            # nn.Conv1d()
            nn.Linear(2 * self.hidden_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1)
        )
    def forward(self, x):
        # The input `x` is a Python list with 2 elements, each a tensor of size
        #   (INPUT_LEN, BATCH_SIZE, EMBEDDING_SIZE).

        # Look up embeddings.
        article_one = x[0]
        article_two = x[1]


        _, out_1 = self.bert_layer(article_one['inputs_ids'], attention_mask = article_one['attention_mask'])
        _, out_2 = self.bert_layer(article_two['inputs_ids'], attention_mask = article_two['attention_mask'])

        # Concatenate the final outputs of each LSTM.
        linear_in = torch.cat((
            out_1.view(self.batch_size, -1),
            out_2.view(self.batch_size, -1)
        ) ,1)

        # Pass the result through a 2-layer NN.
        return self.out_nn(linear_in).view(-1)

# Create model, loss, and optimizer objects.
model = BertBasedBiameseModel(HIDDEN_SIZE, LATENT_DIM, BATCH_SIZE)
loss_func = nn.HingeEmbeddingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}

x_train, y_train = get_data_train()
x_test, y_test = get_data_test()
train_loader = DataLoader()


training_set = Dataset(x_train.keys(), y_train, x_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)
test_set = Dataset(x_test.keys(), y_test, x_test)
training_generator = torch.utils.data.DataLoader(test_set, **params)


for i in range(NUM_EPOCHS):
    sum_loss = 0.0

    for batch_ones, batch_twos, labels in enumerate(training_generator[0]):

        # Create batch input
        device = torch.device()
        first_examples = torch.Tensor(batch_ones).to(device)
        second_examples = torch.Tensor(batch_twos).to(device)
        model_input = [first_examples, second_examples]

        # Initialize model.
        model.hidden = model.init_hidden()

        # Run model and compute loss.
        predictions = model(model_input)
        loss = loss_func(predictions, labels)
        sum_loss += loss.item()

        # Update weights via backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch loss: %s" % sum_loss)


def get_data_train():
    authors_to_files = {}
    for filename in os.listdir('../data/C50/C50train'):
        author = os.path.join('../data/C50/C50train', filename)
        for fname in os.listdir(author):
            a = os.path.join(author, fname)
            print(a, author)
            authors_to_files[author] = authors_to_files.get(author, []) + [a]

    ret = {}
    labels = []
    for i in range(10000):
        if np.rand(1) > 0.6:
            author = authors_to_files[authors_to_files.keys()[np.random.choice(len(authors_to_files.keys()), 1)]]
            a,b = np.random.choice(len(author), 2)
            ret[i] = (author[a], author[b])
            labels[i] = 1
        else:
            aone, atwo = authors_to_files[authors_to_files.keys()[np.random.choice(len(authors_to_files.keys()), 2, replace = False)]]
            a = np.random.choice(len(aone), 1)
            b = np.random.choice(len(atwo), 1)
            ret[i] = (aone[a], atwo[b])
            labels[i] = -1
    return ret, labels


def get_data_test():
    authors_to_files = {}
    for filename in os.listdir('../data/C50/C50test'):
        author = os.path.join('../data/C50/C50test', filename)
        for fname in os.listdir(author):
            a = os.path.join(author, fname)
            print(a, author)
            authors_to_files[author] = authors_to_files.get(author, []) + [a]

    ret = {}
    labels = []
    for i in range(1000):
        if np.rand(1) > 0.6:
            author = authors_to_files[authors_to_files.keys()[np.random.choice(len(authors_to_files.keys()), 1)]]
            a,b = np.random.choice(len(author), 2)
            ret[i] = (author[a], author[b])
            labels[i] = 1
        else:
            aone, atwo = authors_to_files[authors_to_files.keys()[np.random.choice(len(authors_to_files.keys()), 2, replace = False)]]
            a = np.random.choice(len(aone), 1)
            b = np.random.choice(len(atwo), 1)
            ret[i] = (aone[a], atwo[b])
            labels[i] = -1

    return ret, labels


