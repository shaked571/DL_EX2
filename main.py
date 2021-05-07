import argparse
from tagger import MLP, Vocab, DataFile, Trainer, SubWords, MLPSubWords
from preprocessing import TitleProcess


def main(task, part, optimizer, batch_size, l_r, hidden_dim):
    embedding_dim = 50
    word2vec = False
    if part == 3:
        word2vec = True

    vocab = Vocab(task, word2vec)

    sub_words = None
    if part == 4:
        sub_words = SubWords(task)
        model = MLPSubWords(embedding_dim, hidden_dim, vocab, sub_words)
    else:
        model = MLP(embedding_dim, hidden_dim, vocab)

    title_process = TitleProcess()
    train_df = DataFile(task, 'train', title_process, vocab, sub_words)
    dev_df = DataFile(task, 'dev', title_process, vocab, sub_words)
    test_df = DataFile(task, 'test', title_process, vocab, sub_words)

    trainer = Trainer(model=model,
                      train_data=train_df,
                      dev_data=dev_df,
                      vocab=vocab,
                      n_ep=7,
                      optimizer=optimizer,
                      train_batch_size=batch_size,
                      lr=l_r,
                      part=part)
    trainer.train()
    test_prediction = trainer.test(test_df)
    trainer.dump_test_file(test_prediction, test_df.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--optimizer', type=str, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--l_r', type=float, required=False)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.task, args.part, args.optimizer, args.batch_size, args.l_r, args.hidden_dim)
