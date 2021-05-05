import argparse
from tagger import MLP, Vocab, DataFile, Tranier
from preprocessing import TitleProcess


def main(task, part, embedding_dim, batch_size, l_r, hidden_dim):
    if part == 3:
        embedding_dim = 50
    title_process = TitleProcess()
    vocab = Vocab(task)
    train_df = DataFile(task, 'train', title_process, vocab)
    dev_df = DataFile(task, 'dev', title_process, vocab)
    test_df = DataFile(task, 'test', title_process, vocab)
    model = MLP(embedding_dim, hidden_dim, vocab)
    trainer = Tranier(model, train_df, dev_df, vocab, 15, batch_size, l_r)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--embedding_dim', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.task, args.part, args.embedding_dim, args.batch_size, args.l_r, args.hidden_dim)
