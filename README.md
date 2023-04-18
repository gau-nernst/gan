# Generative Adversarial Networks (GANs)

Features:

- GAN losses and regularizations
  - Original GAN loss (non-saturating version)
  - WGAN and WGAN-GP
  - SN-GAN
  - Hinge loss
- Architectures
  - DCGAN
  - NVIDIA GANs:
    - Progressive GAN
    - StyleGAN
    - StyleGAN2 (may implement ADA version)
    - Not implemented (yet): path length regularization
  - Conditional GAN (modified for CNN)
  - SA-GAN

TODO:

- AC-GAN
- BigGAN
- Pix2Pix / CycleGAN

Goodies:

- Mixed precision and distributed training is handled by [Accelerate](https://github.com/huggingface/accelerate)
- Logging with Tensorboard (may add W&B in the future)

## Usage

Train DCGAN on MNIST, using hyperparameters specified in DCGAN paper (image size is fixed at 32x32)

```bash
python gan_mnist.py --model dcgan --method gan --log_name mnist_dcgan --batch_size 32 --n_steps 100000 --lr 2e-4 --beta1 0.5 --beta2 0.999
```

Add `--conditional` for conditional generation (using Conditional GAN. See `gan_mnist.py` for more details)

Unconditional generation | Conditional generation
-------------------------|-----------------------
<video src="https://user-images.githubusercontent.com/26946864/211148684-4e41a917-6459-408f-bd89-e99392ad918a.mp4"> | <video src="https://user-images.githubusercontent.com/26946864/211149361-368e77cb-584b-49fa-9b02-83f175abb422.mp4">

Train DCGAN on CelebA to generate 64x64 images, same hyperparameters as above, with EMA

```bash
python gan_celeba.py --model dcgan --method gan --img_size 64 --log_name celeba_dcgan --batch_size 128 --n_steps 100000 --lr 2e-4 --beta1 0.5 --beta2 0.999 --ema
```

Without EMA | With EMA
------------|---------
<video src="https://user-images.githubusercontent.com/26946864/211149449-0e45259a-ec81-4627-a6dd-6098373a0ee8.mp4"> | <video src="https://user-images.githubusercontent.com/26946864/211149453-770a043d-476c-4d57-8250-26bd9118801c.mp4">

Train DCGAN with WGAN loss on CelebA, using hyperparameters specified in the paper

```bash
python gan_celeba.py --model dcgan --method wgan --img_size 64 --log_name celeba_dcgan_wgan --batch_size 64 --n_steps 100000 --optimizer RMSprop --lr 5e-5 --train_g_interval 5
```

Progressive GAN

```bash
python gan_celeba.py --model progressive_gan --loggers tensorboard --img_size 256 --z_dim 512 --batch_size 16 --method wgan-gp --log_name celeba_progressive_gan --n_steps 650000 --optimizer Adam --beta1 0 --beta2 0.99 --lr 1e-3 --ema --drift_penalty 0.001 --checkpoint_interval 50000
```

### Usage with HF's Accelerate

Run `accelerate config` to create a default configuration. Then you can launch as:

```bash
accelerate launch gan_celeba.py ...
```

Mixed-precision training

```bash
accelerate launch --mixed_precision fp16 gan_celeba.py ...
```

DDP training (w/ mixed-precision) (not so useful since GANs typically cannot use large batch size)

```bash
accelerate launch --num_processes 4 gan_celeba.py ...
```

With `torch.compile()` and mixed-precision training (there are some errors...)

```bash
accelerate launch --mixed_precision fp16 --dynamo_backend inductor gan_celeba.py ...
```

For other options, see `accelerate -h` or [here](https://huggingface.co/docs/accelerate/basic_tutorials/launch).

## Lessons

Some lessons I have learned from implementing and training GANs:

- I could never get MLP-based GANs working. CNN-based GANs are much easier to train.
- Use small batch size e.g. 32 (at least for original GAN. DCGAN paper used 128, WGAN paper used 64). Large batch size will make Discriminator much stronger than Generator. I haven't experimented batch size effect for WGAN, although theoretically WGAN should be able to be trained with large batch (since WGAN wants the best Discriminator given a Generator to estimate the correct Earth mover's distance). BigGAN uses large batch size (according to paper, haven't tested).
- For transposed convolution, use `kernel_size=4` to avoid checkerboard artifacts (used by DCGAN).
- DCGAN's Discriminator does not use pooling at the output, but flatten and matmul to 1 output. This is implemented as convolution with kernel size = feature map size. The paper states it helps with convergence speed. For the Generator, DCGAN applies matmul to latent vector and reshape to (512,4,4). This is implemented as convolution transpose with kernel size = output feature map size i.e. 4. All kernel sizes are 4 in Discriminator and Generator.
- Batch norm helps GANs converge much faster. Regarding which mode (training vs evaluation) Batch norm layer should be at different stages, it seems most implementations leave it in training mode during training.
  - For Discriminator, since it is only used in training, a reasonable choice is for Batch norm to always be in training mode (use current batch statistics).
  - For Generator, Batch norm should be in training mode during training, even when it generates samples for training Discriminator. During inference, if the batch size is large, Batch norm can be in training mode also. Generating single image (batch size 1) might be problematic.
  - I haven't explored other norm layers e.g. Layer norm, Instance norm, Adaptive Instance norm.
- Training dynamics: GAN training depends on random seed, sometimes I can get better (or worse) results by repeating the same training. GAN training may become unstable / collapse after some time, so early stopping is required.
- Generated images can get desaturated / washed-out after a while. It seems like this starts to happen when Discriminator loss becomes plateau. There doesn't seem any literature on this phenomenon.
- WGAN and WGAN-GP:
  - They are not always better than the original GAN loss, while requiring longer training.
  - WGAN: it is necessary to train Discriminator more than Generator (so that Discriminator is a good EMD estimator given a fixed Generator), otherwise Discriminator may collapse `D(x) = D(G(z))`.
- Optimizer: most GANs use Adam with low `beta1` (0.5 or 0.0). Some GANs (WGAN-GP, NVIDIA GANs) require `beta1=0` for stable training. WGAN uses RMSprop. I haven't experimented with other optimizers. SGD probably won't be able to optimize the minimax game. GANs also don't use weight decay.
- Provide label information helps with GAN training. I didn't try modifying Discriminator to classify all classes + fake (suggested by [Salimans 2016](https://proceedings.neurips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)), but Conditional GAN seems to speed up convergence. Conditional GAN probably prevents mode collapse also.
- Progressive GAN:
  - With fp16 mixed precision training, PixelNorm and MinibatchStdDev need to be computed in fp32 for numerical stability.
  - Equalized learning rate helps with training stability. (I have tried not using Equalized LR and scaling LR accordingly but it didn't work. I still think Equalized LR is not necessary since no other networks need that. TODO: re-try again with full fp32 precision)
  - I'm not sure if Discriminator output drift penalty is necessary
  - Mini-batch standard deviation in Discriminator and beta1=0 seem to be important
  - Tanh is not used in Generator (to force values in [-1,1])
  - The author needed to use batch size 16 in order to fit training in GPU memory at 1024x1024 resolution. Having larger batch size is actually better, but hyperparameters need to be adjusted i.e. larger learning rate.
- StyleGAN:
  - Blurring (upfirdn2d) is quite problematic. Using grouped convolution implementation, the 2nd order gradient calculated by PyTorch is extremely slow. This can be fixed by overriding its backward pass (by subclassing `torch.autograd.Function`) with its forward pass (they are identical). WGAN-GP loss in Progressive GAN and R1 regularization in StyleGAN/StyleGAN2 both require 2nd order gradient.
  - Most popular re-implementations like MMGeneration, rosinality, PyTorch-StudioGAN use NVIDIA's custom CUDA kernel (upfirdn2d, introduced in StyleGAN2). The custom kernel is around 1.7x faster (both forward and backward) than grouped convolution implementation at float32 precision (measured on RTX 3090). However, it doesn't support float16 precision. Thus, float16 grouped convolution implementation is faster float32 custom kernel (at least on RTX 3090).
  - More benchmark info can be found at [#1](https://github.com/gau-nernst/gan/issues/1).
- SA-GAN: Generator uses Conditional Batch Norm for conditional generation, which follows [Miyato 2018](https://arxiv.org/abs/1802.05637). BigGAN, which extends SA-GAN, does not change this aspect. There is no evidence yet, but I think Conditional / Adaptive Instance Norm should be better (StyleGAN's approach).
- EMA of the Generator is extremely beneficial. Training DCGAN on CelebA, EMA reduces strange artifacts, makes the generated images smoother and more coherent. [Yazıc 2019](https://arxiv.org/abs/1806.04498) studies this effect. NVIDIA GANs (Progressive GAN, StyleGAN series) and BigGAN use EMA.
