import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--log_name", default="tensorboard")
    return parser
