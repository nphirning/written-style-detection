import os
import pickle as pkl
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch


STEP = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
authors_to_files = pkl.load(open( "a2f.p", "rb" ))

atat = pkl.load(open('a2a_token.p', "rb"))

example = atat['NickLouth'][0]

tokens_tensor = torch.tensor([example])
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = torch.tensor([[0]*512])
segments_tensors = segments_tensors.to('cuda')
model = BertModel.from_pretrained('bert-base-uncased')
model.to('cuda')
model.eval()
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

print ("Number of layers:", len(encoded_layers))
layer_i = 0

print ("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

token_embeddings = torch.stack(encoded_layers, dim=0)
token_embeddings = torch.squeeze(token_embeddings, dim=1)
token_embeddings = token_embeddings.permute(1,0,2)

token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)

print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


pkl.dump(token_vecs_sum, open( "exampleEmbed.p", "wb" ))


# authors_to_articles_string = {}
# authors_to_token = {}

# for author in authors_to_files:
#     print(author)
#     for article in authors_to_files[author]:
#         with open(article, "r") as f:
#             f = f.read()
#             authors_to_articles_string[author] = authors_to_articles_string.get(author, []) + [f]
#             tokenized = tokenizer.tokenize(f)
#             for i in range(0, len(tokenized), STEP):
#                 inds = tokenizer.convert_tokens_to_ids(tokenized[i:i+STEP])
#                 if len(inds) == 512:
#                     authors_to_token[author] = authors_to_token.get(author, []) + [list(inds)]


# pkl.dump(authors_to_articles_string, open( "a2a_string.p", "wb" ))
# pkl.dump(authors_to_token, open( "a2a_token.p", "wb" ))










# authors_to_files = {}

# for filename in os.listdir(path_to_data):
#     author = os.path.join(path_to_data, filename)
#     print(author)
#     for filename in os.listdir(author):
#         a = os.path.join(author, filename)
#         name = a[21:21 + a[21:].index('/')]
#         authors_to_files[name] = authors_to_files.get(name, []) + [a]

# pkl.dump(authors_to_files, open( "a2f.p", "wb" ) )


