import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gan", "wgan", "wgan-gp"], default="gan")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSprop"], default="Adam")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--log_img_interval", type=int, default=2_000)
    parser.add_argument("--log_comment", default="")
    return parser
