import torch
import torch
from transformers import BertTokenizer
import numpy as np

TOKENS = 512


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, list):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.list = list
        self.embeddings_dict = {}
        with open("../data/glove.6B.100d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
  def get_text(self, id):
    with open(id, 'r') as f:
        return f.read()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        id = self.list_IDs[index]
        a,b = self.get_text(self.list[id][0]), self.get_text(self.list[id][1])
        embedded_a = []
        embedded_b = []
        tokens_a = a.split(" ")
        tokens_b = b.split(" ")
        for word in tokens_a:
            e = self.embeddings_dict[word] if word in self.embeddings_dict else [0]*100
            embedded_a.append(e)
        for word in tokens_b:
            e = self.embeddings_dict[word] if word in self.embeddings_dict else [0]*100
            embedded_b.append(e)
        if len(embedded_a) > TOKENS:
            embedded_a = np.array(embedded_a[:512])
        else:
            embedded_a = np.array(embedded_a + ([[0] * 100] * (TOKENS - len(tokens_a))))
        if len(embedded_b) > TOKENS:
            embedded_b = np.array(embedded_b[:512])
        else:
            embedded_b = np.array(embedded_b + ([[0] * 100] * (TOKENS - len(tokens_b))))

        y = self.labels[id]
        return np.mean(embedded_a, axis=0), np.mean(embedded_b, axis=0), y