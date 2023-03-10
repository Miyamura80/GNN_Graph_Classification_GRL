import time
import configparser
import torch
import argparse
import os.path as osp
from utils import get_dataset, get_model
from experiments.run_sc_gc import run_sc_model_gc


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Default Configuration
config = configparser.ConfigParser()
config.read("config.ini")



# CLI configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset", help="Dataset to test the model on.", required=True
)
parser.add_argument("-b", "--batch_size", help="Batch size.", default=32, type=int)
parser.add_argument("-m", "--model", help="The model we will use.", default="GAT")

# Training arguments
parser.add_argument("--lr", help="Learning rate.", default=0.001, type=float)

# Model specific arguments
parser.add_argument(
    "--max_distance", help="Maximal distance in HSP model (K)", default=5, type=int
)
parser.add_argument(
    "--num_layers", help="Number of HSP layers in the model.", default=6, type=int
)
parser.add_argument(
    "--emb_dim", help="Size of the emb dimension.", default=64, type=int
)
parser.add_argument("--scatter", help="Max or Mean pooling.", default="max")
parser.add_argument("--dropout", help="Dropout probability.", default=0.5, type=float)
parser.add_argument("--eps", help="Epsilon in GIN.", default=0.0, type=float)
parser.add_argument("--epochs", help="Number of epochs.", default=500, type=int)
parser.add_argument(
    "--nb_reruns", help="Repeats per task (default 5)", type=int, default=3
)

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = osp.join(osp.dirname(osp.realpath(__file__)), "..")



train_graphs, valid_graphs, num_feat, num_pred = get_dataset(args, root_dir)



model = get_model(
    args,
    device,
    num_features=num_feat,
    num_classes=num_pred,
)


run_sc_model_gc(
    model,
    train_graphs,
    valid_graphs,
    lr=args.lr,
    batch_size=args.batch_size,
    epochs=args.epochs,
)


