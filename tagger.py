from collections import Counter
from datetime import datetime

from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, T_co
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
from preprocessing import PreProcessor
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


class SubWords:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    SUB_WORD_SIZE = 3
    SHORT_SUB_WORD = "SHORT"

    def __init__(self, task: str):
        self.train_path = os.path.join(self.BASE_PATH, task, 'train')
        self.suffix, self.prefix = self.get_prefix_and_suffix()
        self.suffix2i = {s: i for i, s in enumerate(self.suffix)}
        self.i2suffix = {i: s for i, s in enumerate(self.suffix)}
        self.prefix2i = {p: i for i, p in enumerate(self.prefix)}
        self.i2prefix = {i: p for i, p in enumerate(self.prefix)}

    def get_prefix_and_suffix(self):
        suffixes = {self.SHORT_SUB_WORD}
        prefixes = {self.SHORT_SUB_WORD}

        with open(self.train_path) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split("\t")

            if len(word) < self.SUB_WORD_SIZE:
                continue

            suffix = word[len(word) - self.SUB_WORD_SIZE:]
            prefix = word[:self.SUB_WORD_SIZE]
            suffixes.add(suffix)
            prefixes.add(prefix)

        return prefixes, suffixes


class Vocab:
    UNKNOWN_WORD = "UUUNKKK"
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    VOCAB_PATH = os.path.join(BASE_PATH, 'vocab.txt')

    def __init__(self, task: str, word2vec):
        self.word2vec = word2vec
        self.train_path = os.path.join(self.BASE_PATH, task, 'train')
        if self.word2vec:
            _, self.labels = self.get_unique(self.train_path)
            self.words = self.get_word2vec_words()
        else:
            self.words, self.labels = self.get_unique(self.train_path)

        self.vocab_size = len(self.words)
        self.num_of_labels = len(self.labels)
        self.i2word = {i: w for i, w in enumerate(self.words)}
        self.word2i = {w: i for i, w in self.i2word.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}

    def get_word_index(self, word):
        if self.word2vec:
            word = word.lower()

        if word in self.word2i:
            return self.word2i[word]

        return self.word2i[self.UNKNOWN_WORD]

    def get_word2vec_words(self):
        vocab = []
        with open(self.VOCAB_PATH) as f:
            lines = f.readlines()
        for line in lines:
            word = line.strip()
            vocab.append(word)
        return vocab

    def get_unique(self, path):
        words = set()
        labels = set()
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                continue
            word, label = line.strip().split("\t")
            words.add(word)
            labels.add(label)
        words.update(["</s>", "<s>", self.UNKNOWN_WORD])

        return words, labels


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
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    def __init__(self, task: str, data_set, pre_processor: PreProcessor, vocab: Vocab):
        self.data_path = os.path.join(self.BASE_PATH, task, data_set)
        self.pre_processor: PreProcessor = pre_processor
        self.vocab: Vocab = vocab
        self.data: List[InputExample] = self.read_examples_from_file()

    def read_sents(self, lines):
        sentences = []
        sent = []
        for line in lines:
            if line == "" or line == "\n":
                if sent:
                    sentences.append(sent)
                    sent = []
            else:
                sent.append(line.strip().split("\t"))
        return sentences

    def read_examples_from_file(self):
        guid_index = 1
        sent_index = 0
        examples = []
        with open(self.data_path, encoding="utf-8") as f:
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
                examples.append(InputExample(guid=f"{self.data_path}-{guid_index}-{sent_index}", words=words, label=labels[2]))
                sent_index += 1
            guid_index += 1
        return examples

    def __getitem__(self, index) -> T_co:
        words = self.data[index].words
        label = self.data[index].label
        words_tensor = torch.tensor([self.vocab.get_word_index(w) for w in words]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label]]).to(torch.int64)

        return words_tensor, label_tensor

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):
    PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'wordVectors.txt')

    def __init__(self, embedding_size: int, hidden_dim: int, vocab: Vocab):
        super(MLP, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.vocab_size = self.vocab.vocab_size
        self.embed_dim = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim) #

        # init embedding using word2vec
        if self.vocab.word2vec:
            weights = np.loadtxt(self.PATH)
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.linear1 = nn.Linear(self.embed_dim * 5, self.hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_dim, self.vocab.num_of_labels)

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)

        return out


class Trainer:
    def __init__(self, model: nn.Module, train_data: DataFile, dev_data: DataFile,
                 vocab: Vocab,
                 n_ep=1,
                 train_batch_size=8,
                 steps_to_eval=4000,
                 lr=0.01):
        self.model = model
        self.dev_batch_size = 128
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        self.dev_data = DataLoader(dev_data, batch_size=self.dev_batch_size,)

        self.vocab = vocab
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_args = {"lr": lr,
                           "epoch": self.n_epochs,
                           "batch_size": train_batch_size,
                           "steps_to_eval": self.steps_to_eval
                           }
        self.writer = SummaryWriter(log_dir=f"tensor_board/{self.suffix_run()}")

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
            step_loss = 0
            self.model.train() # prep model for training
            for step, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                data = data.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data) # Eemnded Data Tensor size (1,5)
                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item()*data.size(0)
                step_loss += loss.item()*data.size(0)
                if step % 4000 == 0:
                    print(f"in step: {step} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    self.evaluate_model(step * (epoch + 1), "step")

            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.evaluate_model(epoch, "epoch")

    def evaluate_model(self, step, stage):
        self.model.eval()
        loss = 0

        prediction = []
        all_target = []
        for eval_step, (data, target) in tqdm(enumerate(self.dev_data), total=len(self.dev_data),
                                              desc=f"dev step {step} loop"):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)

            loss = self.loss_func(output, target.view(-1))
            loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            prediction += predicted.tolist()
            all_target += target.view(-1).tolist()
        accuracy = self.accuracy_token_tag(prediction, all_target)
        print(f'Accuracy/dev_{stage}: {accuracy}')
        self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy,step)
        self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
        self.model.train()

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self, test_df):
        self.model.eval()
        prediction = []
        for test_step, (data, target) in tqdm(enumerate(test_df), total=len(test_df),
                                              desc=f"test data"):
            data = data.to(self.device)
            output = self.model(data)
            _, predicted = torch.max(output, 1)
            prediction += predicted.tolist()
        return [self.vocab.i2label[i] for i in prediction]
        

    def accuracy_token_tag(self, predict: List, target: List):
        predict = [self.vocab.i2label[i] for i in predict]
        target = [self.vocab.i2label[i] for i in target]
        all_pred = 0
        correct = 0
        for p, t in zip(predict, target):
            if t == 'O' and p == 'O':
                continue
            all_pred += 1
            if t == p:
                correct += 1
        return (correct / all_pred) * 100

    def dump_test_file(self, test_prediction, test_file_path):
        res = []
        cur_i = 0
        with open(test_file_path) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                res.append(line)
            else:
                pred = f"{line}\t{test_prediction[cur_i]}\n"
                res.append(pred)
                cur_i += 1
        pred_path = f"{self.suffix_run()}.tsv"
        with open(pred_path, mode='w') as f:
            f.writelines(res)




# if __name__ == '__main__':
#     title_process = TitleProcess()
#     vocab = Vocab(os.path.join('data','ner'))
#     # train_df = DataFile(os.path.join('data','ner'), 'train', title_process, vocab)
#     train_df = DataFile('data/ner/train', title_process, vocab)
#     dev_df = DataFile('data/ner/dev', title_process,  vocab)
#     test_df = DataFile('data/ner/test', title_process,  vocab)
#     model = MLP(50, 100, vocab) #define how the network looks like
#     # trainer = Tranier(model, train, 2)
#     trainer = Tranier(model, train_df, dev_df,  vocab, 10)
#     trainer.train()
