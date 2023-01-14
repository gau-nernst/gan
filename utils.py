import argparse
import inspect
from typing import Literal, Union, get_args, get_origin

from training import GANTrainerConfig


def add_args_from_cls(parser: argparse.ArgumentParser, cls):
    for k, v in inspect.signature(cls).parameters.items():
        arg_type = v.annotation

        if arg_type in (int, float, str):
            parser.add_argument(f"--{k}", type=arg_type, default=v.default)

        elif arg_type is bool:
            assert v.default is False
            parser.add_argument(f"--{k}", action="store_true")

        elif get_origin(arg_type) is Literal:
            literals = get_args(arg_type)
            assert all(isinstance(literal, str) for literal in literals)
            parser.add_argument(f"--{k}", default=v.default, choices=literals)

        # only for Optional[]
        elif get_origin(arg_type) is Union:
            args = get_args(arg_type)
            assert len(args) == 2 and args[1] is type(None)
            assert args[0] in (int, float, str)
            parser.add_argument(f"--{k}", type=args[0], default=v.default)

        else:
            raise ValueError(f"Unsupported type {arg_type}")


def cls_from_args(args: argparse.Namespace, cls):
    kwargs = {k: getattr(args, k) for k in inspect.signature(cls).parameters.keys()}
    return cls(**kwargs)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dcgan")
    parser.add_argument("--base_depth", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--n_log_imgs", type=int, default=40)
    add_args_from_cls(parser, GANTrainerConfig)
    return parser
