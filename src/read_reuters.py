import os
import pickle as pkl
from pytorch_pretrained_bert import BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
authors_to_files = pkl.load(open( "a2f.p", "rb" ))

authors_to_articles_string = {}
authors_to_articles_token = {}

for author in authors_to_files:
    print(author)
    for article in authors_to_files[author]:
        print(article)
        with open(article, "r") as f:
            f = f.read()
            authors_to_articles_string[author] = authors_to_articles_string.get(author, []) + [f]
            tokenized = tokenizer.tokenize(f)
            authors_to_articles_token[author] = authors_to_articles_token.get(author, []) + [tokenized]


pkl.dump(authors_to_articles_string, open( "a2a_string.p", "wb" ))
pkl.dump(authors_to_articles_token, open( "a2a_token.p", "wb" ))










# authors_to_files = {}

# for filename in os.listdir(path_to_data):
#     author = os.path.join(path_to_data, filename)
#     print(author)
#     for filename in os.listdir(author):
#         a = os.path.join(author, filename)
#         name = a[21:21 + a[21:].index('/')]
#         authors_to_files[name] = authors_to_files.get(name, []) + [a]

# pkl.dump(authors_to_files, open( "a2f.p", "wb" ) )


