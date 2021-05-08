import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse

class MostSimilar:
    def __init__(self, vectors_path, vocab_path, k):
        self.vectors = self.get_vectors(vectors_path)
        self.word2i, self.i2word = self.get_vocab(vocab_path)
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


    def most_similar(self, word):
        word_vec = self.vectors[self.word2i[word]]
        def cosine_on_w( v):
            return dot(word_vec, v) / (norm(word_vec) * norm(v))
        cosine_all_words = np.apply_along_axis(cosine_on_w, axis=1, arr=self.vectors)
        cosine_with_index = [(k, v) for k, v in enumerate(cosine_all_words)]
        cosine_with_index.sort(key=lambda x: x[1], reverse=True)
        top_k_words = [(self.i2word[ind], cosine) for ind, cosine in cosine_with_index[1:self.k+1]] # not to take the first word
        return top_k_words


def main(vectors_path, vocab_path, req_word, k, all_words):
    ms = MostSimilar(vectors_path, vocab_path, k)
    if all_words:
        words = "dog england john explode office".split()
    else:
        words = [req_word]
    for word in words:
        k_most_similar = ms.most_similar(word)
        print("*"*40)
        print("word:")
        print(word)
        for sim_word, cosine_val in k_most_similar:
            print(f"sim word: {sim_word} cosine val: {cosine_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--vectors_path', type=str, default='data/wordVectors.txt', required=False)
    parser.add_argument('--vocab_path', type=str, default='data/vocab.txt', required=False)
    parser.add_argument('--word', type=str, required=False)
    parser.add_argument('--k', type=int, required=False)
    parser.add_argument('-a','--all' ,action='store_true')

    args = parser.parse_args()

    main(args.vectors_path, args.vocab_path, args.word, args.k,  args.all)
