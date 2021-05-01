import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse


class MostSimilar:
    def __init__(self, vectors_path, vocab_path, word, k):
        self.vectors = self.get_vectors(vectors_path)
        self.word2i, self.i2word = self.get_vocab(vocab_path)
        self.word_vec = self.vectors[self.word2i[word]]
        self.k = k

    def get_vectors(self, vectors_path):
        return np.loadtxt(vectors_path)

    def get_vocab(self, vocab_path):
        word2i = {}
        with open(vocab_path) as f:
            words = f.readlines()
            for i, word in enumerate(words):
                word2i[word.strip()] = i
        i2word = {v: k for k, v in word2i.items()}
        return word2i, i2word

    def cosine(self, v):
        return dot(self.word_vec, v) / (norm(self.word_vec) * norm(v))

    def most_similar(self):
        cosine_all_words = np.apply_along_axis(self.cosine, axis=1, arr=self.vectors)
        cosine_with_index = [(k, v) for k, v in enumerate(cosine_all_words)]
        cosine_with_index.sort(key=lambda x: x[1], reverse=True)
        top_k_ind = [ind for ind, cosine in cosine_with_index[:self.k]]
        top_k_words = [self.i2word[ind] for ind in top_k_ind]
        return top_k_words


def main(vectors_path, vocab_path, word, k):
    ms = MostSimilar(vectors_path, vocab_path, word, k)
    k_most_similar = ms.most_similar()
    for word in k_most_similar:
        print(word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--vectors_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--word', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)

    args = parser.parse_args()

    main(args.vectors_path, args.vocab_path, args.word, args.k)
