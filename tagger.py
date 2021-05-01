import torch
from torch import nn
import numpy as np
from  preprocessing import LowerProcess, PreProcessor
import os
# Read the files

# Calculate Precision\Recall in NER and POS (count only tag != 'O')

# writing a cosine function
# retrieval of most similar. (in the top_k.py file)

# tecnique
# Class of word embedding-with diffrent init options
# Different schemes of representing words  -with a flag -
#                                           pretrained vec
#                                           random init
#                                           a flag with or without subwords
# How does the input looks like?

# Differnt schemes of training - { MLP + size of matrix + batch size (hyper paraeter - is there nore?)
#                                 CNN + num of filters window size}
#                               3.1 Building the network architecture  MLP+TanH+Softmax
#                               3.2 Building the CNN architecture - Max pooling, filters etc.
# saving all the loss to a tensorboardX + Adduracy and Recall
# Add a function that create a file name for each run
#
#

class Parser:
    def __init__(self, path: str, pre_processor: PreProcessor):
        self.path = path
        self.data = self.read_file(self.path)
        self.
        self.vocab =
        self.tags =
        self.i2word =
        self.word2i =

    def read_file(self, path):
        pass


class MLP(nn.Module):
    PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__), 'data','wordVectors.txt')) #TODO maybe give an option flag

    def __init__(self, embedding_size,vocab_size, hidden_dim, load_embedding=False):
        super(MLP, self).__init__()

        self.hidden_dim  = hidden_dim
        if load_embedding:
            self.embed_dim = 50
            self.vocab_size = 0 # TODO from read embeddinf
            #code to load embedinng
            #init embediing using load
            weights = np.loadtxt(self.PATH)
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.embedding.weight.data.copy_(torch.from_numpy(weights))
        else:
            self.embed_dim = embedding_size
            #init embediing using randomly
            self.vocab_size = 0 # TODO
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        self.layers = nn.Sequential(
                    self.embedding,
                    nn.Linear(self.embed_dim, self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, self.unique_tags)
                )

    def forward(self, x):

        output = self.embedding(x)




        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x



class MLP:
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    def __init__(self, embedding_size, vocab_size,load_embedding=False):#TODO what are the flgs?

        self.loss = nn.CrossEntropyLoss()
        self.mlp1 = nn.MulLP



    def model(self):