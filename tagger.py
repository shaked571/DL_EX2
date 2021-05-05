from collections import Counter
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple, T_co
from torch.utils.tensorboard import SummaryWriter

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
class Vocab:
    UNKNOWN_WORD = "UUUNKKK"

    def __init__(self, dir_path: str):
        self.train_path = os.path.join(dir_path, "train")
        self.words, self.labels = self.get_unique(self.train_path)
        self.vocab_size = len(self.words)
        self.num_of_labels = len(self.labels)
        self.i2word = {i: w for i, w in enumerate(self.words)}    # TODO make the indexes according to the vocab.txt file
        self.word2i = {w: i for i, w in self.i2word.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def get_unique(self, path):
        words = set()
        labels = set()
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                continue
            word, label = line.strip().split("\t")
            words.add(word)
            labels.add(label)
        words.update(["</s>", "<s>", self.UNKNOWN_WORD])

        return words, labels


    # model.get_embed_vector(data)


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

    def __init__(self, data_dir: str, mode: str,  pre_processor: PreProcessor, vocab: Vocab):
        self.data_dir = data_dir
        self.mode: str = mode
        self.pre_processor: PreProcessor = pre_processor
        self.vocab: Vocab = vocab
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

    def get_word_index(self, word):
        if word in self.vocab.word2i:
            return self.vocab.word2i[word]
        return self.vocab.word2i[self.vocab.UNKNOWN_WORD]

    def __getitem__(self, index) -> T_co:
        words = self.data[index].words
        label = self.data[index].label
        words_tensor = torch.Tensor([self.get_word_index(w) for w in words]).to(torch.int64)
        label_tensor = torch.Tensor([self.vocab.label2i[label]]).to(torch.int64)

        return words_tensor, label_tensor

    def __len__(self):
        return len(self.data)



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

            self.linear1 = nn.Linear(self.embed_dim * 5, self.hidden_dim)
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(self.hidden_dim, self.vocab.num_of_labels)


    # def get_embed_vector(self, window):
    #     window_indexes = [vocab.word2i[word[0]] for word in window]
    #     lookup_tensor = torch.tensor(window_indexes, dtype=torch.long)
    #     embed_window = self.embedding(lookup_tensor)
    #     return embed_window
    #
    #
    # def get_embed_vector(self, window):
    #     embed_window = []
    #     for word in window:
    #         # window_indexes = [vocab.word2i[word[0]] for word in window]
    #         lookup_tensor = torch.tensor([vocab.word2i[word[0]]], dtype=torch.long)
    #         embed_window.append(self.embedding(lookup_tensor))
    #     embed_window = torch.cat(tuple(embed_window), 1)
    #     return embed_window

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)

        return out

    # def transform_input(self, input: List[List[str]])-> torch.Tensor:


class Tranier:
    # def __init__(self, model: nn.Module, data: DataFile, n_ep, ): #optimizer,lr
    def __init__(self, model: nn.Module, train_data: DataFile, dev_data: DataFile,
                 vocab: Vocab,
                 n_ep=1,
                 train_batch_size=4,
                 dev_batch_size=8,
                 to_shuffle=True):
        self.model = model
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, shuffle=to_shuffle)
        self.dev_data = DataLoader(dev_data, batch_size=dev_batch_size,)
        self.vocab = vocab
        self.optimizer = optim.SGD(model.parameters(), lr=0.01)

        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=self.get_trainer_name())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def train(self):
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf # set initial "min" to infinity
        # monitor training loss
        total_train_loss = 0.0
        total_dev_loss = 0.0

        for epoch in range(self.n_epochs):
            ###################
            # train the model #
            ###################
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0

            self.model.train() # prep model for training
            for i, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                data = data.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data) # Eemnded Data Tensor size (1,5)
                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item()*data.size(0)
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            # self.writer.add_scalar('Accuracy/train', np.random.random(), n_iter)

        #  TODO add  here a full run on dev set.
            #  TODO save model states with epoch i. togethere with loss
            #  TODO add loss dev, loss train, accuracy to the tensorboard
            loss_dev = 0
            model.eval()
            for i, (data, target) in enumerate(self.train_data):
                data = data.to(self.device)
                output = model(data) # Eemnded Data Tensor size (1,5)
                loss = self.loss_func(output, target.view(-1))
                loss_dev += loss.item()*data.size(0)
            self.writer.add_scalar('Loss/dev', loss_dev, epoch)
        # self.writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


    def get_trainer_name(self):
        return "FFFFFFFFFFFFFFFFIXXX MEEEEe" # TODO


if __name__ == '__main__':
    title_process = TitleProcess()
    vocab = Vocab(os.path.join('data','ner'))
    train_df = DataFile(os.path.join('data','ner'), 'train', title_process, vocab)
    dev_df = DataFile(os.path.join('data','ner'), 'dev', title_process,  vocab)
    test_df = DataFile(os.path.join('data','ner'), 'test', title_process,  vocab)
    model = MLP(50, 100, vocab) #define how the network looks like
    # trainer = Tranier(model, train, 2)
    trainer = Tranier(model, train_df, dev_df,  vocab, 2)
    trainer.train()
