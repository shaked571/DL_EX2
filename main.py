import argparse
from tagger import MLP, Vocab, DataFile, Trainer, SubWords, MLPSubWords
from preprocessing import TitleProcess


def main(task, part, embedding_dim,optimizer, batch_size, l_r, hidden_dim):
    word2vec = False
    sub_words = None
    if part == 3:
        embedding_dim = 50
        word2vec = True

    vocab = Vocab(task, word2vec)

    if part == 4:
        sub_words = SubWords(task)
        model = MLPSubWords(embedding_dim, hidden_dim, vocab, sub_words)
    else:
        model = MLP(embedding_dim, hidden_dim, vocab)

    title_process = TitleProcess()
    train_df = DataFile(task, 'train', title_process, vocab, sub_words)
    dev_df = DataFile(task, 'dev', title_process, vocab, sub_words)
    test_df = DataFile(task, 'test', title_process, vocab, sub_words)

    trainer = Trainer(model, train_df, dev_df, vocab, 5, optimizer, batch_size, l_r)
    trainer.train()
    test_prediction = trainer.test(test_df)
    trainer.dump_test_file(test_prediction, test_df.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--embedding_dim', type=int, required=False)
    parser.add_argument('--optimizer', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.task, args.part, args.embedding_dim, args.optimizer,args.batch_size, args.l_r, args.hidden_dim)
