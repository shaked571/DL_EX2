import argparse
from tagger import MLP, Vocab, DataFile, Tranier
from preprocessing import TitleProcess


def main(train_path, dev_path, test_path, part, embedding_dim, batch_size, l_r, hidden_dim):
    vocab_from_train = True
    if part == 3:
        embedding_dim = 50
        vocab_from_train = False

    title_process = TitleProcess()
    vocab = Vocab(train_path, vocab_from_train)
    train_df = DataFile(train_path, title_process, vocab)
    dev_df = DataFile(dev_path, title_process, vocab)
    test_df = DataFile(test_path, title_process, vocab)
    model = MLP(embedding_dim, hidden_dim, vocab)
    trainer = Tranier(model, train_df, dev_df, vocab, 15, batch_size, l_r)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--embedding_dim', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.train_path, args.dev_path, args.test_path, args.part, args.embedding_dim, args.batch_size, args.l_r, args.hidden_dim)
