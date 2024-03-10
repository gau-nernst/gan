# Generative Adversarial Networks (GANs)

Features:

- GAN losses:
  - [Original GAN loss](https://arxiv.org/abs/1406.2661) (non-saturating version i.e. replace `log(1-sigmoid(d(x)))` with `logsigmoid(-d(x))`)
  - [WGAN](https://arxiv.org/abs/1701.07875)
  - [Hinge loss](https://arxiv.org/abs/1802.05957)
  - [LSGAN](https://arxiv.org/abs/1611.04076)
  - [Relativistic GAN](https://arxiv.org/abs/1807.00734)
- GAN regularization:
  - [WGAN-GP](https://arxiv.org/abs/1704.00028)
  - [R1](https://arxiv.org/abs/1801.04406)
- Others:
  - [DiffAuugment](https://arxiv.org/abs/2006.10738)
- Architectures
  - [DCGAN](https://arxiv.org/abs/1511.06434)
  - Conditional GAN (modified for CNN) (TODO: remove)
  - NVIDIA GANs:
    - (Modified) [Progressive GAN](https://arxiv.org/pdf/1710.10196)
    - (Modified) [StyleGAN](https://arxiv.org/abs/1812.04948)
    - (Modified) [StyleGAN2](https://arxiv.org/abs/1912.04958)
  - [SA-GAN](https://arxiv.org/pdf/1805.08318)
- Img2Img:
  - [Pix2Pix](https://arxiv.org/abs/1611.07004) / [CycleGAN](https://arxiv.org/abs/1703.10593): PatchGAN discriminator, Unet and ResNet generator
    - Dropout not included. Batch norm is replaced with Instance norm.
    - Not implemented: train Discrimiantor with past generated samples
  - TODO: Pix2PixHD, AnimeGAN, SRGAN, ESRGAN, StarGAN

TODO:

- StyleGAN2-ADA
- AC-GAN
- BigGAN

Notes in train script:

- For mixed precision training, only bf16 is supported (so that I don't need to use gradient scaler).
- No multi-GPU support.

## Env setup

Tested with PyTorch 2.2. Older PyTorch should work too.

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pytorch-fid tqdm wandb
```

## Usage

### CelebA 64x64

By default, the generator is trained for 30k iterations. EMA is always used. FID is calculated using 10k samples.

Model | Loss | Batch size | Command | FID | Note | Samples
------|------|------------|---------|-----|------|--------
DCGAN | Original GAN | 128 | `python train_celeba.py --run_name dcgan --lr 2e-5 --optimizer_kwargs '{"betas":[0.5,0.999]}' --batch_size 128 --mixed_precision --channels_last` | 51.66 | |
DCGAN | WGAN | 64 | `python train_celeba.py --run_name dcgan_wgan --lr 5e-5 --optimizer RMSprop --batch_size 64 --n_disc 5 --method wgan --mixed_precision --channels_last` | 26.16 | |
DCGAN | WGAN-GP | 64 | `python train_celeba.py --run_name dcgan_wgangp --disc_kwargs '{"norm":"none"}' --lr 1e-4 --optimizer_kwargs '{"betas":[0,0.9]}' --batch_size 64 --n_disc 5 --method wgan --regularizer wgan-gp --mixed_precision --channels_last` | 17.26 | No bn in discriminator. |
DCGAN | Hinge | 64 | `python train_celeba.py --run_name dcgan_sngan --disc_kwargs '{"norm":"none"}' --sn_disc --lr 1e-4 --optimizer_kwargs '{"betas":[0.5,0.999]}' --batch_size 64 --method hinge --mixed_precision --channels_last` | 23.10 | **SN-GAN**. No bn in discriminator. Spectral norm in discriminator. Original SN-GAN did not use DCGAN architecture. |
DCGAN | Relativistic GAN | 64 | `python train_celeba.py --run_name dcgan_rgan --disc_kwargs '{"norm":"none"}' --sn_disc --lr 2e-4 --optimizer_kwargs '{"betas":[0.5,0.999]}' --batch_size 64 --method relativistic-gan --mixed_precision --channels_last` | 20.34 | No bn in discriminator. Spectral norm in discriminator. |
SAGAN | Hinge | 256 | `python train_celeba.py --run_name sagan --model sagan --sn_disc --sn_gen --lr 2e-4 --optimizer_kwargs '{"betas":[0,0.9]}' --batch_size 256 --method hinge --mixed_precision --channels_last` | 6.73 | Spectral norm in discriminator and generator. Original SAGAN uses different learning rates for Generator (1e-4) and Discriminator (4e-4).  |
Progressive GAN | Hinge | 64 | `python train_celeba.py --run_name progan_hinge --model progressive_gan --disc_kwargs '{"base_dim":64}' --sn_disc --gen_kwargs '{"base_dim":64}' --sn_gen --lr 2e-4 --optimizer_kwargs '{"betas":[0,0.9]}' --batch_size 64 --method hinge --mixed_precision --channels_last` | | No progressive growing and equalized learning rate like in the original. Use residual discriminator and generator (similar to StyleGAN2/SA-GAN). Trained with SA-GAN hyperparameters (Hinge loss, spectral norm for both discriminator and generator). |
StyleGAN | Hinge | 64 | `python train_celeba.py --run_name stylegan_hinge_r1 --model stylegan --disc_kwargs '{"base_dim":64}' --gen_kwargs '{"base_dim":64}' --lr 2e-4 --optimizer_kwargs '{"betas":[0,0.9]}' --batch_size 64 --method hinge --regularizer r1 --mixed_precision --channels_last` | |

### CelebA 256x256

DCGAN with Relativistic GAN loss

![dcgan_celeba256_rgan](https://github.com/gau-nernst/gan/assets/26946864/9deaeb5d-c618-45a6-96d7-a3572ef52ba9)

Progressive GAN with Hinge loss and spectral norm in Discriminator

![dcgan_celeba256_progran_resDresG_hinge](https://github.com/gau-nernst/gan/assets/26946864/a71647e7-c5e9-422f-9b0d-3bbfe402a6c9)

### Pix2Pix and CycleGAN

By default, dropout=0 since I don't see any benefits of using them.

```bash
python train_pix2pix.py --dataset cityscapes --mixed_precision --channels_last --compile --optimizer_kwargs '{"betas":[0.5,0.999]}' --run_name cityscapes
python train_cyclegan.py --dataset horse2zebra --mixed_precision --channels_last --compile --optimizer_kwargs '{"betas":[0.5,0.999]}' --run_name horse2zebra
```

## Lessons

General

- I could never get MLP-based GANs working. CNN-based GANs are much easier to train.
- Most GANs don't scale with larger batch size. Typically, a small batch size like 32-64 is good (DCGAN paper used 128, WGAN paper used 64). Large batch size will make Discriminator much stronger than Generator.
  - I haven't experimented batch size effect for WGAN, although theoretically WGAN should be able to be trained with large batch (since WGAN wants the best Discriminator given a Generator to estimate the correct Earth mover's distance). (future note: doesn't really make sense).
  - SA-GAN (CNN with self-attention + Hinge loss) can scale well with batch size (Table 1 in [BigGAN](https://arxiv.org/abs/1809.11096), consistent improvements from batch size 256 to 2048, though it is not a fair comparison).
- For upsampling with transposed convolution (`stride=2`), use `kernel_size=4` to avoid checkerboard artifacts (used by DCGAN).
- DCGAN's Discriminator does not use pooling at the output, but flatten and matmul to 1 output. This is implemented as convolution with kernel size = feature map size. The paper states it helps with convergence speed. For the Generator, DCGAN applies matmul to latent vector and reshape to (512,4,4). This is implemented as convolution transpose with kernel size = output feature map size i.e. 4. All kernel sizes are 4 in Discriminator and Generator.
- Batch norm helps GANs converge much faster. Regarding which mode (training vs evaluation) Batch norm layer should be at different stages, it seems most implementations leave it in training mode during training.
  - For Discriminator, since it is only used in training, a reasonable choice is for Batch norm to always be in training mode (use current batch statistics).
  - For Generator, Batch norm should be in training mode during training, even when it generates samples for training Discriminator. During inference, if the batch size is large, Batch norm can be in training mode also. Generating single image (batch size 1) might be problematic.
  - I haven't explored other norm layers e.g. Layer norm, Instance norm, Adaptive Instance norm.
- Training dynamics: GAN training depends on random seed, sometimes I can get better (or worse) results by repeating the same training. GAN training may become unstable / collapse after some time, so early stopping is required.
- Generated images can get desaturated / washed-out after a while. It seems like this starts to happen when Discriminator loss becomes plateau. There doesn't seem any literature on this phenomenon. (update: This was likely a bug in my previous implementation. If the discriminator has batch norm after its first convolution layer, it won't be able to tell the saturation in its input images. Removing the batch norm, as stated in DCGAN paper, seems to resolve this issue.)
- Optimizer: most GANs use Adam with low `beta1` (0.5 or 0.0). Some GANs (WGAN-GP, NVIDIA GANs) require `beta1=0` for stable training. WGAN uses RMSprop (not momentum-based). Momentum does not make sense for optimizing GANs, since the objective is not stationary. GANs also don't use weight decay.
- Provide label information helps with GAN training. It is probably beneficial for multi-modal distributions, thus learning a mapping from N-d Gaussian to data distribution is easier. There are several approaches to this:
  - Conditional GAN (CGAN)
  - Make Discriminator classify all classes + fake (suggested by [Salimans 2016](https://proceedings.neurips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html))
  - AC-GAN
- EMA of the Generator is extremely beneficial. Training DCGAN on CelebA, EMA reduces strange artifacts, makes the generated images smoother and more coherent. [Yazıc 2019](https://arxiv.org/abs/1806.04498) studies this effect. NVIDIA GANs (Progressive GAN, StyleGAN series) and BigGAN use EMA.

WGAN and WGAN-GP:

- It is necessary to train Discriminator more than Generator (so that Discriminator is a good EMD estimator given a fixed Generator), otherwise Discriminator may collapse `D(x) = D(G(z))`.
- They are not always better than the original GAN loss, while requiring longer training (at least 2x slower). Calculate 2nd order derivative is also expensive (for WGAN-GP).

Relativistic GAN:

- Simple, fast, and excellent results. It beats WGAN, WGAN-GP, and SN-GAN, while having the speed of original GAN. TODO: see if RGAN can scale well with batch size.
- Without spectral norm, discriminator output will keep either increasing or decreasing.
- For more complicated networks like Progressive GAN and StyleGAN architecture, especially at higher resolutions, Relativistic GAN cannot converge.

Progressive GAN:

- With fp16 mixed precision training, PixelNorm and MinibatchStdDev need to be computed in fp32 for numerical stability. This can be done simply by calling `.float()` inside `.forward()` (no-op if input is already fp32).
- IMPORTANT: When using gradient penalty (WGAN-GP or R1) with MinibatchStdDev and mixed precision, division by zero might occur in backward pass when std is 0 (underflow). To avoid this, I have to use a custom implementation of std, with an addition of eps before performing sqrt().
- Equalized learning rate does not seem to be important. SA-GAN, with a similar architecture, can be trained noramlly.
- Discriminator output drift penalty is not really necessary. It helps stabilize training at the start, but doesn't improve the results later in training.
- Mini-batch standard deviation in Discriminator and beta1=0 seem to be important
- Tanh is not used in Generator (to force values in [-1,1])
- The author needed to use batch size 16 in order to fit training in GPU memory at 1024x1024 resolution. Having larger batch size is actually better, but hyperparameters need to be adjusted i.e. larger learning rate.

StyleGAN:

- Blurring (upfirdn2d) is quite problematic. Using grouped convolution implementation, the 2nd order gradient calculated by PyTorch is extremely slow. This can be fixed by overriding its backward pass (by subclassing `torch.autograd.Function`) with its forward pass (they are identical). WGAN-GP loss in Progressive GAN and R1 regularization in StyleGAN/StyleGAN2 both require 2nd order gradient.
- Most popular re-implementations like MMGeneration, rosinality, PyTorch-StudioGAN use NVIDIA's custom CUDA kernel (upfirdn2d, introduced in StyleGAN2). The custom kernel is around 1.7x faster (both forward and backward) than grouped convolution implementation at float32 precision (measured on RTX 3090). However, it doesn't support float16 precision. Thus, float16 grouped convolution implementation is faster float32 custom kernel (at least on RTX 3090).
- More benchmark info can be found at [#1](https://github.com/gau-nernst/gan/issues/1).
- StyleGAN is very hard to train. Without some kind of gradient penalty (R1 like in original paper, or WGAN-GP penalty like in [lucidrains' implementation](https://github.com/lucidrains/stylegan2-pytorch)), the model cannot learn at all.

SA-GAN:

- Generator uses Conditional Batch Norm for conditional generation, which follows [Miyato 2018](https://arxiv.org/abs/1802.05637). BigGAN, which extends SA-GAN, does not change this aspect. There is no evidence yet, but I think Conditional / Adaptive Instance Norm should be better (StyleGAN's approach).

Pix2Pix:

- Removing dropout yields better generated images. With dropout, the images are noisy and have strange texture (Discriminator seems to collapse). Therefore, seeing Pix2Pix as conditional GAN is not quite right. It's directly learning a mapping from one domain to another, trained with GAN objective. Whether and why this works better than direct training with supervised objective is not clear to me.
- Edges2shoes dataset: there are some anti-aliasing.

SRGAN:

- Conv3x3 + Pixel shuffle is equivalent to ConvTranspose6x6. From my benchmarks, ConvTranspose is faster most of the time.
