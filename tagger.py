from dataclasses import dataclass
from typing import List, Optional, Tuple, T_co

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
from  preprocessing import LowerProcess, PreProcessor, TitleProcess
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

#                                 CNN + num of filters window size}
#                               3.1 Building the network architecture  MLP+TanH+Softmax
#                               3.2 Building the CNN architecture - Max pooling, filters etc.
# saving all the loss to a tensorboardX + Adduracy and Recall
# Add a function that create a file name for each run
#
#


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        label: (Optional) str. The label of the middle word in the window
    """
    guid: str
    words: List[str]
    label: Optional[str]


class DataFile(Dataset):
    WINDOW_SIZE = 5

    def __init__(self, data_dir: str, mode: str,  pre_processor: PreProcessor):
        self.data_dir = data_dir
        self.mode = mode
        self.pre_processor = pre_processor
        self.data: List[InputExample] = self.read_examples_from_file()

    def read_sents(self, lines):
        sentences = []
        sent = []
        for line in lines:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if sent:
                    sentences.append(sent)
                    sent = []
            else:
                sent.append(line.strip().split("\t"))
        return sentences

    def read_examples_from_file(self):
        file_path = os.path.join(self.data_dir, self.mode)
        guid_index = 1
        sent_index = 0
        examples = []
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        sents = self.read_sents(lines)
        for sent in sents:
            if len(sent[0]) == 1:
                sent = [["<s>"], ["<s>"]] + sent + [["</s>"], ["</s>"]]
            else:
                sent = [["<s>", 'O'], ["<s>", 'O']] + sent + [["</s>", 'O'], ["</s>", 'O']]

            for i in range(len(sent) - self.WINDOW_SIZE + 1):
                batch = sent[i:i+self.WINDOW_SIZE]
                words = []
                labels = []
                for word_label in batch:
                    words.append(self.pre_processor.process(word_label[0]))
                    if len(word_label) > 1:
                        labels.append(word_label[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
                examples.append(InputExample(guid=f"{self.mode}-{guid_index}-{sent_index}", words=words, label=labels[2]))
                sent_index += 1
            guid_index += 1
        return examples

    def __getitem__(self, index) -> T_co:
        return self.data[index].words, self.data[index].label

    def __len__(self):
        return len(self.data)


class Vocab:
    UNKNOWN_WORD = "UUUNKKK"

    def __init__(self, train_file: DataFile):
        self.data_file = train_file
        self.words, self.labels = self.unique_words(self.data_file)
        self.vocab_size = len(self.words)
        self.num_of_labels = len(self.labels)
        self.i2word = {i: w for i, w in enumerate(self.words)}    # TODO make the indexes according to the vocab.txt file
        self.word2i = {w: i for i, w in self.i2word.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def unique_words(self, data_file: DataFile) -> Tuple[set, set]:
        uniq_words = set()
        uniq_labels = set()
        for example in data_file.data:
            uniq_words.update(example.words)
            uniq_labels.add(example.label)
        # Adding the unknown word
        uniq_words.add(self.UNKNOWN_WORD)
        return uniq_words, uniq_labels


class MLP(nn.Module):
    PATH = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data', 'wordVectors.txt') #TODO maybe give an option flag

    def __init__(self, embedding_size: int, hidden_dim: int, vocab: Vocab, load_embedding=False):
        super(MLP, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
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
            self.vocab_size = self.vocab.vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        self.layers = nn.Sequential(
                    # self.embedding, # TODO should we remove it?
                    nn.Linear(self.embed_dim*5, self.hidden_dim), # TODO the dimension updated
                    nn.Tanh(),
                    nn.Linear(self.hidden_dim, self.vocab.num_of_labels)
                )

    # def get_embed_vector(self, window):
    #     window_indexes = [vocab.word2i[word[0]] for word in window]
    #     lookup_tensor = torch.tensor(window_indexes, dtype=torch.long)
    #     embed_window = self.embedding(lookup_tensor)
    #     return embed_window

    def get_embed_vector(self, window):
        embed_window = []
        for word in window:
            # window_indexes = [vocab.word2i[word[0]] for word in window]
            lookup_tensor = torch.tensor([vocab.word2i[word[0]]], dtype=torch.long)
            embed_window.append(self.embedding(lookup_tensor))
        embed_window = torch.cat(tuple(embed_window), 1)
        return embed_window

    def forward(self, x):
        # output = self.embedding(x)
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = self.get_embed_vector(x).to(torch.long)
        # x = self.get_embed_vector(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    # def transform_input(self, input: List[List[str]])-> torch.Tensor:


class Tranier:
    # def __init__(self, model: nn.Module, data: DataFile, n_ep, ): #optimizer,lr
    def __init__(self, model: nn.Module, vocab: Vocab, n_ep, ): #optimizer,lr
        self.model = model
        self.n_epochs = n_ep
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)
        # self.data_loader = DataLoader(data, batch_size=1)
        self.data_loader = DataLoader(vocab.data_file, batch_size=1)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self):
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf # set initial "min" to infinity
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        for epoch in range(self.n_epochs):
            ###################
            # train the model #
            ###################
            print(f"start epoch: {epoch + 1}")
            self.model.train() # prep model for training
            for i, (data, target) in enumerate(self.data_loader):
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                embed_data = model.get_embed_vector(data)
                output = model(embed_data)
                # calculate the loss
                loss = self.loss_func(output, torch.tensor([vocab.label2i[target[0]]]))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item()*embed_data.size(0)
                if i % 1000 == 0:
                    print(f"example number: {i + 1}")
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")


if __name__ == '__main__':
    pp = TitleProcess()
    train = DataFile(os.path.join('data','ner'), 'train', pp)
    dev = DataFile(os.path.join('data','ner'), 'dev', pp)
    test = DataFile(os.path.join('data','ner'), 'test', pp)
    vocab = Vocab(train)
    model = MLP(50, 100, vocab)
    # trainer = Tranier(model, train, 2)
    trainer = Tranier(model, vocab, 2)
    trainer.train()
