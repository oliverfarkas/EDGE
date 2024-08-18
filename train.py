from args import parse_train_opt
from EDGE import EDGE
import os

def train(opt):
    model = EDGE(opt.feature_type)
    model.train_loop(opt)


if __name__ == "__main__":
    os.environ['EDGE_CHECKPOINT'] = "baseline.e50.b32.pt"
    opt = parse_train_opt()
    train(opt)
