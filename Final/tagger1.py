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
from math import sqrt


WINDOW_CONTEXT = 5
torch.manual_seed(1)


class CharsVocab:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    PADDING = "PAD"
    UNKNOWN_CHAR = "UNKNOWN_CHAR"
    WORD_LEN = 20

    def __init__(self, task: str):
        self.separator = " " if task == "pos" else "\t"
        self.train_path = os.path.join(self.BASE_PATH, task, 'train')
        self.chars = self.get_chars()

        self.chars_num = len(self.chars)

        self.char2i = {c: i for i, c in enumerate(self.chars)}
        self.i2char = {i: c for i, c in enumerate(self.chars)}

    def get_chars_indexes_by_word(self, word):
        word_chars = [c for c in word]
        indexes = []

        # add chars indexes
        for c in word_chars[:self.WORD_LEN]:
            if c in self.char2i:
                indexes.append(self.char2i[c])
            else:
                indexes.append(self.char2i[self.UNKNOWN_CHAR])

        # add padding indexes
        for i in range(self.WORD_LEN - len(indexes)):
            indexes.append(self.char2i[self.PADDING])

        return indexes

    def get_chars(self):
        chars = {self.PADDING, self.UNKNOWN_CHAR}
        with open(self.train_path) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split(self.separator)
            chars.update([c for c in word])

        return chars



class SubWords:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    SUB_WORD_SIZE = 3
    SHORT_SUB_WORD = "SHORT_WORD"
    UNKNOWN_SUB_WORD = "UNKNOW_SUB_WORD"

    def __init__(self, task: str):
        self.separator = " " if task == "pos" else "\t"
        self.train_path = os.path.join(self.BASE_PATH, task, 'train')
        self.suffix, self.prefix = self.get_prefix_and_suffix()

        self.suffix_num = len(self.suffix)
        self.prefix_num = len(self.prefix)

        self.suffix2i = {s: i for i, s in enumerate(self.suffix)}
        self.i2suffix = {i: s for i, s in enumerate(self.suffix)}
        self.prefix2i = {p: i for i, p in enumerate(self.prefix)}
        self.i2prefix = {i: p for i, p in enumerate(self.prefix)}

    def get_sub_words_indexes_by_word(self, word):
        if len(word) < self.SUB_WORD_SIZE:
            return self.prefix2i[self.SHORT_SUB_WORD], self.suffix2i[self.SHORT_SUB_WORD]

        prefix, suffix = self.get_sub_words_by_word(word)
        if prefix not in self.prefix2i:
            prefix = self.UNKNOWN_SUB_WORD
        if suffix not in self.suffix2i:
            suffix = self.UNKNOWN_SUB_WORD

        return self.prefix2i[prefix], self.suffix2i[suffix]

    def get_sub_words_by_word(self, word):
        suffix = word[len(word) - self.SUB_WORD_SIZE:]
        prefix = word[:self.SUB_WORD_SIZE]
        return prefix, suffix

    def get_prefix_and_suffix(self):
        suffixes = {self.SHORT_SUB_WORD, self.UNKNOWN_SUB_WORD}
        prefixes = {self.SHORT_SUB_WORD, self.UNKNOWN_SUB_WORD}

        with open(self.train_path) as f:
            lines = f.readlines()

        for line in lines:
            if line == "" or line == "\n":
                continue
            word, _ = line.strip().split(self.separator)

            if len(word) < self.SUB_WORD_SIZE:
                continue

            prefix, suffix = self.get_sub_words_by_word(word)
            suffixes.add(suffix)
            prefixes.add(prefix)

        return prefixes, suffixes


class Vocab:
    UNKNOWN_WORD = "UUUNKKK"
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    VOCAB_PATH = os.path.join(BASE_PATH, 'vocab.txt')

    def __init__(self, task: str, word2vec):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.word2vec = word2vec
        self.train_path = os.path.join(self.BASE_PATH, self.task, 'train')
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
            word, label = line.strip().split(self.separator)
            words.add(word)
            labels.add(label)
        words.update(["</s>", "<s>", self.UNKNOWN_WORD])
        labels.add('O')

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

    def __init__(self, task: str, data_set, pre_processor: PreProcessor, vocab: Vocab,
                 sub_words: SubWords = None,
                 char_vocab: CharsVocab = None):
        self.task = task
        self.separator = " " if self.task == "pos" else "\t"
        self.data_path = os.path.join(self.BASE_PATH, task, data_set)
        self.pre_processor: PreProcessor = pre_processor
        self.vocab: Vocab = vocab
        self.sub_words = sub_words
        self.char_vocab = char_vocab
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
                sent.append(line.strip().split(self.separator))
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
                batch = sent[i:i + self.WINDOW_SIZE]
                words = []
                labels = []
                for word_label in batch:
                    words.append(self.pre_processor.process(word_label[0]))
                    if len(word_label) > 1:
                        labels.append(word_label[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
                examples.append(
                    InputExample(guid=f"{self.data_path}-{guid_index}-{sent_index}", words=words, label=labels[2]))
                sent_index += 1
            guid_index += 1
        return examples

    def get_sub_words_tensor(self, words):
        words_prefixes = []
        words_suffixes = []
        for w in words:
            prefix, suffix = self.sub_words.get_sub_words_indexes_by_word(w)
            words_prefixes.append(prefix)
            words_suffixes.append(suffix)
        prefixes_tensor = torch.tensor(words_prefixes).to(torch.int64)
        suffixes_tensor = torch.tensor(words_suffixes).to(torch.int64)
        return prefixes_tensor, suffixes_tensor

    def __getitem__(self, index) -> T_co:
        words = self.data[index].words
        label = self.data[index].label
        words_tensor = torch.tensor([self.vocab.get_word_index(w) for w in words]).to(torch.int64)
        label_tensor = torch.tensor([self.vocab.label2i[label]]).to(torch.int64)

        if self.sub_words:
            prefixes_tensor, suffixes_tensor = self.get_sub_words_tensor(words)
            words_tensor = torch.stack((words_tensor, prefixes_tensor, suffixes_tensor), dim=0)

        elif self.char_vocab:
            chars_tensor = self.get_chars_tensor(words)
            words_tensor = torch.cat([chars_tensor, words_tensor.repeat(1)[:, None]], axis=1)

        return words_tensor, label_tensor

    def get_chars_tensor(self, words):
        chars_tensor = []  # 20 (num of chars in each word)* 5 (num of words) = 100
        for word in words:
            chars_indices = self.char_vocab.get_chars_indexes_by_word(word)
            chars_tensor.append(chars_indices)
        chars_tensor = torch.tensor(chars_tensor).to(torch.int64)
        return chars_tensor

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
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

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


    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(checkpoint)




class MLPSubWords(MLP):
    def __init__(self, embedding_size: int, hidden_dim: int, vocab: Vocab, sub_words: SubWords):
        super().__init__(embedding_size, hidden_dim, vocab)
        self.sub_words = sub_words
        self.prefix_embedding = nn.Embedding(self.sub_words.prefix_num, self.embed_dim)
        self.suffix_embedding = nn.Embedding(self.sub_words.suffix_num, self.embed_dim)

    def forward(self, x):
        out_word = self.embedding(x[torch.arange(x.size(0)), 0])
        out_pre = self.prefix_embedding(x[torch.arange(x.size(0)), 1])
        out_suf = self.suffix_embedding(x[torch.arange(x.size(0)), 2])
        out = torch.stack((out_word, out_pre, out_suf), dim=0).sum(axis=0)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out


class CnnMLPSubWords(MLP):
    def __init__(self, embedding_size: int, hidden_dim: int, vocab: Vocab,
                 char_embed_dim: int,
                 filter_num: int,
                 window_size: int,
                 char_vocab: CharsVocab):

        super().__init__(embedding_size, hidden_dim, vocab)
        self.char_vocab = char_vocab
        self.char_embed_dim = char_embed_dim
        self.filter_num = filter_num
        self.word_len = self.char_vocab.WORD_LEN
        self.window_size = window_size
        self.conv_feature_len = self.word_len - self.window_size + 1
        self.output_dim = self.char_embed_dim * self.filter_num
        self.linear_dims = {3: 315, 4: 310, 5: 305}
        self.char_embeddings = nn.Embedding(self.char_vocab.chars_num, self.char_embed_dim,
                                            padding_idx=self.char_vocab.char2i[self.char_vocab.PADDING])
        self.char_embeddings.weight.data.uniform_(-sqrt(3/self.char_embed_dim), sqrt(3/self.char_embed_dim))
        self.conv = nn.Conv2d(in_channels=WINDOW_CONTEXT, out_channels=self.filter_num * WINDOW_CONTEXT, kernel_size=self.window_size, stride=1, padding=2, groups=WINDOW_CONTEXT)
        # self.conv = nn.Conv2d(in_channels=WINDOW_CONTEXT, out_channels=self.filter_num, kernel_size=self.window_size, stride=1)
        self.relu = nn.LeakyReLU()
        self.max_pool = nn.MaxPool3d(self.word_len, stride=(self.filter_num, self.window_size, 1))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.linear_dims[self.window_size], self.hidden_dim)

    def forward(self, x):
        # x size is (batch * 5 *21)
        words_tensor = x[:, :, -1]  # words_tensor size is (batch * 5 * 1 )
        out_word = self.embedding(words_tensor)

        chars_tensor = x[:, :, :-1]  # chars_tensor size is (batch * 5 * 20 )
        out_chars = self.char_embeddings(chars_tensor)
        out_chars = self.conv(out_chars)
        out_chars = self.relu(out_chars)
        out_chars = self.max_pool(out_chars).transpose(2, 3)
        out_chars = torch.squeeze(out_chars, 3)
        out = torch.cat((out_chars, out_word), 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.tanh(out)
        out = self.linear2(out)
        return out


class Trainer:

    def __init__(self, model: nn.Module, train_data: DataFile, dev_data: DataFile, vocab: Vocab, n_ep=1,
                 optimizer='AdamW', train_batch_size=8, steps_to_eval=30000, lr=0.01, filter_num=30, window_size=3, part=None):
        self.part = part
        self.model = model
        self.dev_batch_size = 128
        self.train_data = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        self.dev_data = DataLoader(dev_data, batch_size=self.dev_batch_size, )
        self.vocab = vocab
        if optimizer == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer supports SGD, Adam, AdamW")
        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_args = {"part": self.part, "task": self.vocab.task, "lr": lr, "epoch": self.n_epochs,
                           "batch_size": train_batch_size, "filter_num": filter_num, "window_size": window_size,
                           "steps_to_eval": self.steps_to_eval, "optim": optimizer, "hidden_dim": self.model.hidden_dim}
        self.writer = SummaryWriter(log_dir=f"tensor_board/{self.suffix_run()}")

        self.saved_model_path = f"{self.suffix_run()}.bin"

        self.best_model = None
        self.best_score = 0

    def train(self):
        for epoch in range(self.n_epochs):
            ###################
            # train the model #
            ###################
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0
            step_loss = 0
            self.model.train()  # prep model for training
            for step, (data, target) in tqdm(enumerate(self.train_data), total=len(self.train_data)):
                data = data.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)  # Eemnded Data Tensor size (1,5)
                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)
                step_loss += loss.item() * data.size(0)
                if step % self.steps_to_eval == 0:
                    print(f"in step: {step} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    self.evaluate_model(step * (epoch + 1), "step")
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.evaluate_model(epoch, "epoch")

    def evaluate_model(self, step, stage):
        with torch.no_grad():
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
            self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy, step)
            self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
            if accuracy > self.best_score:
                self.best_score = accuracy
                torch.save(self.model.state_dict(), self.saved_model_path)

        self.model.train()

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self, test_df):
        test = DataLoader(test_df, batch_size=self.dev_batch_size, )
        self.model.load_state_dict(torch.load(self.saved_model_path))
        self.model.eval()
        prediction = []
        for test_step, (data, _) in tqdm(enumerate(test), total=len(test), desc=f"test data"):
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
                pred = f"{line.strip()}{self.vocab.separator}{test_prediction[cur_i]}\n"
                res.append(pred)
                cur_i += 1
        pred_path = f"{self.suffix_run()}.tsv"
        with open(pred_path, mode='w') as f:
            f.writelines(res)

