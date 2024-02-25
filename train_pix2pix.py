from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from tqdm import tqdm

from diff_augment import rand_int, translate
from losses import get_loss, get_regularizer
from modelling import build_discriminator, build_generator
from utils import EMA, apply_spectral_norm, cycle, make_parser, normalize, unnormalize


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def run_cmd(cmd: str):
    import shlex
    import subprocess

    subprocess.run(shlex.split(cmd), check=True)


# https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
class Pix2PixDataset(Dataset):
    def __init__(self, root: str, dataset: str, split: str) -> None:
        super().__init__()
        assert dataset in ("cityscapes", "edges2handbags", "edges2shoes", "facades", "maps", "night2day")
        data_dir = Path(root) / dataset

        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            save_path = data_dir / f"{dataset}.tar.gz"
            run_cmd(f'wget https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset}.tar.gz -O "{save_path}"')
            run_cmd(f'tar -xzf "{save_path}" -C "{data_dir}"')

        self.data_dir = data_dir / dataset / split
        self.files = sorted(x.name for x in self.data_dir.iterdir())
        self.split = split

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = io.read_image(str(self.data_dir / self.files[idx]), mode=io.ImageReadMode.RGB)
        A, B = normalize(img).chunk(2, 2)
        if self.split == "train":
            if rand_int(0, 2) == 1:  # random horizontal flip
                A = A.flip(-1)
                B = B.flip(-1)
            if rand_int(0, 2) == 1:  # random translate
                H, W = A.shape[1:]
                translate_x = rand_int(-(W // 5), W // 5 + 1)
                translate_y = rand_int(-(H // 5), H // 5 + 1)
                A = translate(A, translate_x, translate_y)
                B = translate(B, translate_x, translate_y)
        return A, B

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class TrainConfig:
    model: str = "pix2pix"
    dataset: str = "none"
    disc_kwargs: dict = field(default_factory=dict)
    gen_kwargs: dict = field(default_factory=dict)
    sn_disc: bool = False
    sn_gen: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    channels_last: bool = False
    compile: bool = False
    grad_accum: int = 1
    n_iters: int = 30_000
    lr: float = 2e-4
    optimizer: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    batch_size: int = 16
    method: str = "gan"
    regularizer: str = "none"

    run_name: str = "pix2pix"
    log_img_interval: int = 1_000


if __name__ == "__main__":
    args = make_parser(TrainConfig).parse_args()
    cfg = TrainConfig(**vars(args))

    print("Config:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    disc = build_discriminator(cfg.model, **cfg.disc_kwargs).to(cfg.device)
    gen = build_generator(cfg.model, **cfg.gen_kwargs).to(cfg.device)
    disc.apply(apply_spectral_norm) if cfg.sn_disc else None
    gen.apply(apply_spectral_norm) if cfg.sn_gen else None
    if cfg.channels_last:
        gen.to(memory_format=torch.channels_last)
        disc.to(memory_format=torch.channels_last)
    if cfg.compile:
        gen.compile()
        disc.compile()

    print(disc)
    print(gen)

    gen_ema = EMA(gen)
    criterion = get_loss(cfg.method)
    regularizer = get_regularizer(cfg.regularizer)

    optim_cls = getattr(torch.optim, cfg.optimizer)
    optim_d = optim_cls(disc.parameters(), cfg.lr, **cfg.optimizer_kwargs)
    optim_g = optim_cls(gen.parameters(), cfg.lr, **cfg.optimizer_kwargs)

    ds = Pix2PixDataset("data", cfg.dataset, "train")
    dloader = DataLoader(
        ds, cfg.batch_size // cfg.grad_accum, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    dloader = cycle(dloader)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    val_ds = Pix2PixDataset("data", cfg.dataset, "test" if cfg.dataset == "night2day" else "val")
    fixed_As = []
    fixed_Bs = []
    for i in range(0, len(val_ds), (len(val_ds) + 99) // 100):
        A, B = val_ds[i]
        fixed_As.append(A)
        fixed_Bs.append(B)
    fixed_As = torch.stack(fixed_As, 0)
    fixed_Bs = torch.stack(fixed_Bs, 0).to(cfg.device)

    logger = wandb.init(project="pix2pix", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    reals = fixed_As.unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
    io.write_png(unnormalize(reals), f"{log_img_dir}/reals.png")

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters, dynamic_ncols=True)
    while step < cfg.n_iters:
        for _ in range(cfg.grad_accum):
            As, Bs = next(dloader)
            As = As.to(cfg.device)
            Bs = Bs.to(cfg.device)

            if cfg.channels_last:
                As = As.to(memory_format=torch.channels_last)
                Bs = Bs.to(memory_format=torch.channels_last)

            with autocast_ctx:
                with torch.no_grad():
                    fakes = gen(Bs)
                d_reals = disc(As, Bs)
                d_fakes = disc(fakes, Bs)
                loss_d = criterion.d_loss(d_reals, d_fakes)
            loss_d.backward()
        optim_d.step()
        optim_d.zero_grad()

        disc.requires_grad_(False)
        for i in range(cfg.grad_accum):
            with autocast_ctx:
                fakes = gen(Bs)
                loss_g = criterion.g_loss(disc(fakes, Bs), lambda: disc(As, Bs))
                loss_g += F.l1_loss(As, fakes) * 100  # content loss
            loss_g.backward()
        optim_g.step()
        optim_g.zero_grad()
        disc.requires_grad_(True)
        gen_ema.step()

        step += 1
        pbar.update()
        if step % 50 == 0:
            log_dict = {
                "loss/d": loss_d.item(),
                "d/real": d_reals.detach().mean().item(),
                "d/fake": d_fakes.detach().mean().item(),
                "loss/g": loss_g.item(),
            }
            logger.log(log_dict, step=step)

        if step % cfg.log_img_interval == 0:
            for suffix, model in [("", gen), ("_ema", gen_ema)]:
                with autocast_ctx, torch.no_grad():
                    fakes = model(fixed_Bs)
                fakes = fakes.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
                fakes = unnormalize(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")
