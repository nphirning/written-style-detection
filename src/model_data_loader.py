import torch
import torch
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, list):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.list = list

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

        ID = self.list[id]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded_dict_a = tokenizer.encode_plus(
                        self.get_text(ID[0]),                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        encoded_dict_b = tokenizer.encode_plus(
                        self.get_text(ID[1]),                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        y = self.labels[id]
        return encoded_dict_a, encoded_dict_b, y