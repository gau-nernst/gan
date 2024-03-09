from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.transforms import v2
from tqdm import tqdm

from losses import get_loss, get_regularizer
from modelling import build_discriminator, build_generator
from utils import EMA, cycle, make_parser, normalize_img, prepare_model, unnormalize_img


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def run_cmd(cmd: str):
    import shlex
    import subprocess

    subprocess.run(shlex.split(cmd), check=True)


# http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/
class CycleGanDataset(Dataset):
    _datasets = (
        "ae_photos",
        "apple2ornage",
        "cezane2photo",
        "cityscapes",
        "facades",
        "grumpifycat",
        "horse2zebra",
        "iphone2dslr_flower",
        "maps",
        "mini",
        "mini_colorization",
        "mini_pix2pix",
        "monet2photo",
        "summer2winter_yosemite",
        "ukiyoe2photo",
        "vangogh2photo",
    )

    def __init__(self, root: str, dataset: str, split: str) -> None:
        super().__init__()
        assert dataset in self._datasets
        data_dir = Path(root) / "cyclegan" / dataset

        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            save_path = data_dir / f"{dataset}.zip"
            run_cmd(f'wget http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{dataset}.zip -O "{save_path}"')
            run_cmd(f'unzip "{save_path}" -d "{data_dir}"')

        self.data_dir = data_dir / dataset / split
        self.files = sorted(x.name for x in self.data_dir.iterdir())
        self.split = split
        self.transform = v2.RandomHorizontalFlip() if split.startswith("train") else None

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = io.read_image(str(self.data_dir / self.files[idx]), mode=io.ImageReadMode.RGB)
        img = normalize_img(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.files)


@dataclass
class TrainConfig:
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

    run_name: str = "cyclegan"
    log_img_interval: int = 1_000


if __name__ == "__main__":
    args = make_parser(TrainConfig).parse_args()
    cfg = TrainConfig(**vars(args))

    print("Config:")
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")

    disc1 = build_discriminator("cyclegan", **cfg.disc_kwargs).to(cfg.device)
    disc2 = build_discriminator("cyclegan", **cfg.disc_kwargs).to(cfg.device)
    gen1 = build_generator("cyclegan", **cfg.gen_kwargs).to(cfg.device)
    gen2 = build_generator("cyclegan", **cfg.gen_kwargs).to(cfg.device)

    prepare_model(disc1, spectral_norm=cfg.sn_disc, channels_last=cfg.channels_last, compile=cfg.compile)
    prepare_model(disc2, spectral_norm=cfg.sn_disc, channels_last=cfg.channels_last, compile=cfg.compile)
    prepare_model(gen1, spectral_norm=cfg.sn_gen, channels_last=cfg.channels_last, compile=cfg.compile)
    prepare_model(gen2, spectral_norm=cfg.sn_gen, channels_last=cfg.channels_last, compile=cfg.compile)

    print(disc1)
    print(gen1)

    gen1_ema = EMA(gen1)
    gen2_ema = EMA(gen2)
    criterion = get_loss(cfg.method)
    regularizer = get_regularizer(cfg.regularizer)

    optim_cls = getattr(torch.optim, cfg.optimizer)
    optim_d = optim_cls(list(disc1.parameters()) + list(disc2.parameters()), cfg.lr, **cfg.optimizer_kwargs)
    optim_g = optim_cls(list(gen1.parameters()) + list(gen2.parameters()), cfg.lr, **cfg.optimizer_kwargs)

    train_ds_A = CycleGanDataset("data", cfg.dataset, "trainA")
    train_ds_B = CycleGanDataset("data", cfg.dataset, "trainB")
    dloader_A = DataLoader(
        train_ds_A, cfg.batch_size // cfg.grad_accum, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    dloader_B = DataLoader(
        train_ds_B, cfg.batch_size // cfg.grad_accum, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    dloader_A = cycle(dloader_A)
    dloader_B = cycle(dloader_B)

    autocast_ctx = torch.autocast(cfg.device, dtype=torch.bfloat16, enabled=cfg.mixed_precision)
    val_ds_A = CycleGanDataset("data", cfg.dataset, "testA")
    val_ds_B = CycleGanDataset("data", cfg.dataset, "testB")
    fixed_As = []
    fixed_Bs = []
    for i in range(100):
        fixed_As.append(val_ds_A[i])
        fixed_Bs.append(val_ds_B[i])
    fixed_As = torch.stack(fixed_As, 0).to(cfg.device)
    fixed_Bs = torch.stack(fixed_Bs, 0).to(cfg.device)

    logger = wandb.init(project="cyclegan", name=cfg.run_name, config=vars(cfg))
    log_img_dir = Path("images_cyclegan") / cfg.run_name
    log_img_dir.mkdir(parents=True, exist_ok=True)

    real_As = fixed_As.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
    real_Bs = fixed_Bs.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
    io.write_png(unnormalize_img(real_As), f"{log_img_dir}/reals_1.png")
    io.write_png(unnormalize_img(real_Bs), f"{log_img_dir}/reals_2.png")

    step = 0  # generator update step
    pbar = tqdm(total=cfg.n_iters, dynamic_ncols=True)
    while step < cfg.n_iters:
        for _ in range(cfg.grad_accum):
            As = next(dloader_A).to(cfg.device)
            Bs = next(dloader_B).to(cfg.device)

            if cfg.channels_last:
                As = As.to(memory_format=torch.channels_last)
                Bs = Bs.to(memory_format=torch.channels_last)

            with autocast_ctx:
                with torch.no_grad():
                    fake_As = gen1(Bs)
                    fake_Bs = gen2(As)
                loss_d1 = criterion.d_loss(disc1(As, Bs), disc1(fake_As, Bs))
                loss_d2 = criterion.d_loss(disc2(As, Bs), disc1(As, fake_Bs))
                loss_d = loss_d1 + loss_d2
            loss_d.backward()
        optim_d.step()
        optim_d.zero_grad()

        disc1.requires_grad_(False)
        disc2.requires_grad_(False)
        for i in range(cfg.grad_accum):
            with autocast_ctx:
                fake_As = gen1(Bs)
                fake_Bs = gen2(As)
                loss_g = (
                    criterion.g_loss(disc1(fake_As, Bs), lambda: disc1(fake_As, Bs))
                    + F.l1_loss(fake_As, As) * 100
                    + criterion.g_loss(disc2(As, fake_Bs), lambda: disc2(As, fake_Bs))
                    + F.l1_loss(fake_Bs, Bs) * 100
                )
            loss_g.backward()
        optim_g.step()
        optim_g.zero_grad()
        disc1.requires_grad_(True)
        disc2.requires_grad_(True)
        gen1_ema.step()
        gen2_ema.step()

        step += 1
        pbar.update()
        if step % 50 == 0:
            log_dict = {
                "loss/d": loss_d.item(),
                "loss/g": loss_g.item(),
            }
            logger.log(log_dict, step=step)

        if step % cfg.log_img_interval == 0:
            for suffix, model, xs in [
                ("_1", gen1, fixed_Bs),
                ("_ema1", gen1_ema, fixed_Bs),
                ("_2", gen2, fixed_As),
                ("_ema2", gen2_ema, fixed_Bs),
            ]:
                with autocast_ctx, torch.no_grad():
                    fakes = model(xs)
                fakes = fakes.cpu().unflatten(0, (10, 10)).permute(2, 0, 3, 1, 4).flatten(1, 2).flatten(-2, -1)
                fakes = unnormalize_img(fakes)
                io.write_png(fakes, f"{log_img_dir}/step{step // 1000:04d}k{suffix}.png")
