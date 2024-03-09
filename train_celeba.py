from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, io
from torchvision.transforms import v2
from tqdm import tqdm

from diff_augment import DiffAugment
from fid import FID
from losses import get_loss, get_regularizer
from modelling import build_discriminator, build_generator
from utils import EMA, cycle, make_parser, prepare_model, unnormalize_img


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    model: str = "dcgan"
    img_size: int = 64
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
    n_disc: int = 1
    lr: float = 2e-4
    optimizer: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    batch_size: int = 64
    method: str = "gan"
    regularizer: str = "none"
    diff_augment: bool = False

    run_name: str = "dcgan_celeba"
    log_img_interval: int = 1_000
    fid_interval: int = 5_000


if __name__ == "__main__":
    args = make_parser(TrainConfig).parse_args()
    cfg = TrainConfig(**vars(args))

    print("Config:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    disc = build_discriminator(cfg.model, img_size=cfg.img_size, **cfg.disc_kwargs).to(cfg.device)
    gen = build_generator(cfg.model, img_size=cfg.img_size, **cfg.gen_kwargs).to(cfg.device)

    disc = nn.Sequential(DiffAugment(), disc) if cfg.diff_augment else disc

    prepare_model(disc, spectral_norm=cfg.sn_disc, channels_last=cfg.channels_last, compile=cfg.compile)
    prepare_model(gen, spectral_norm=cfg.sn_gen, channels_last=cfg.channels_last, compile=cfg.compile)

    print(disc)
    print(gen)

    gen_ema = EMA(gen)
    criterion = get_loss(cfg.method)
    regularizer = get_regularizer(cfg.regularizer)

    optim_cls = getattr(torch.optim, cfg.optimizer)
    optim_d = optim_cls(disc.parameters(), cfg.lr, **cfg.optimizer_kwargs)
    optim_g = optim_cls(gen.parameters(), cfg.lr, **cfg.optimizer_kwargs)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(cfg.img_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(cfg.img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = datasets.CelebA("data", transform=transform, download=True)
    dloader = DataLoader(
        ds, cfg.batch_size // cfg.grad_accum, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    dloader = cycle(dloader)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    fixed_zs = torch.randn(100, 128, device=cfg.device)

    fid_scorer = FID(cfg.device)
    celeba_stats_path = Path(f"celeba{cfg.img_size}_stats.pth")

    if not celeba_stats_path.exists():
        celeba_stats = fid_scorer.compute_stats(lambda: next(dloader)[0].to(cfg.device))
        torch.save(celeba_stats, celeba_stats_path)

    else:
        celeba_stats = torch.load(celeba_stats_path, map_location="cpu")

    logger = wandb.init(project="celeba", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images_celeba") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters, dynamic_ncols=True)
    while step < cfg.n_iters:
        for i in range(cfg.n_disc):
            cached_reals = [] if i == cfg.n_disc - 1 else None

            if cfg.method == "wgan" and cfg.regularizer == "none":
                with torch.no_grad():
                    for p in disc.parameters():
                        p.clip_(-0.01, 0.01)

            for _ in range(cfg.grad_accum):
                reals, _ = next(dloader)
                reals = reals.to(cfg.device)
                if cfg.channels_last:
                    reals = reals.to(memory_format=torch.channels_last)
                if cached_reals is not None:
                    cached_reals.append(reals.clone())  # cached for generator later
                reals.requires_grad_()

                zs = torch.randn(reals.shape[0], 128, device=cfg.device)
                with autocast_ctx:
                    with torch.no_grad():
                        fakes = gen(zs)
                    d_reals = disc(reals)
                    d_fakes = disc(fakes)
                    loss_d = criterion.d_loss(d_reals, d_fakes)
                    if regularizer is not None:
                        loss_d += regularizer(disc, reals, fakes, d_reals)
                loss_d.backward()

            optim_d.step()
            optim_d.zero_grad()

        disc.requires_grad_(False)
        for i in range(cfg.grad_accum):
            zs = torch.randn(reals.shape[0], 128, device=cfg.device)
            with autocast_ctx:
                loss_g = criterion.g_loss(disc(gen(zs)), lambda: disc(cached_reals[i]))
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
                    fakes = model(fixed_zs)
                fakes = fakes.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
                fakes = unnormalize_img(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")

        if step % cfg.fid_interval == 0:

            def closure():
                zs = torch.randn(cfg.batch_size, 128, device=cfg.device)
                with autocast_ctx, torch.no_grad():
                    return gen_ema(zs).float()

            stats = fid_scorer.compute_stats(closure)
            fid_score = fid_scorer.fid_score(celeba_stats, stats)
            logger.log({"fid/ema": fid_score}, step=step)
