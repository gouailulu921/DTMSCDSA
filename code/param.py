import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MAGCNSE.")
    parser.add_argument("--seed",
                        nargs="?",
                        default=50,
                        help="Random seed")
    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../data",
                        help="Data.")
    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of training epochs for representation learning. Default is 150.")
    parser.add_argument("--gcn-layers",
                        type=int,
                        default=4,
                        help="Number of GCN Layers. Default is 4.")
    parser.add_argument("--out-channels",
                        type=int,
                        default=128,
                        help="out-channels of CNN. Default is 256.")
    parser.add_argument("--circRNA-number",
                        type=int,
                        default=271,
                        help="circRNA number. Default is 271.")
    parser.add_argument("--fc",
                        type=int,
                        default=128,
                        help="circRNA feature dimensions. Default is 256.")
    parser.add_argument("--drug-number",
                        type=int,
                        default=218,
                        help="drug number. Default is 218.")
    parser.add_argument("--fd",
                        type=int,
                        default=128,
                        help="drug feature dimensions. Default is 256.")
    parser.add_argument("--circRNA-view",
                        type=int,
                        default=1,
                        help="circRNA views number. Default is 1(1 datasets for circRNA sim)")
    parser.add_argument("--drug-view",
                        type=int,
                        default=1,
                        help="drug views number. Default is 1(1 datasets for drug sim)")

    return parser.parse_args()
