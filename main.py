import argparse
from tagger import MLP


def main(train_path, test_path, part, batch_size, hidden_dim):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)

    args = parser.parse_args()

    main(args.train_path, args.test_path, args.part, args.batch_size, args.hidden_dim)
