from tokenizers import BertWordPieceTokenizer
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for article in reuters:
    tokenize(article)
    #done
print(tokenizer.encode(s))
print(tokenizer.decode(1010))
# print(tokenizer.encode(s))
