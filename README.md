# Generative Adversarial Networks (GANs)

Features:

- GAN losses and regularizations
  - Original GAN loss (non-saturating version)
  - WGAN and WGAN-GP
  - SN-GAN
  - Hinge loss
- Architectures
  - DCGAN
  - NVIDIA GANs (WIP): Progressive GAN, StyleGAN, StyleGAN2
  - Conditional GAN
  - SA-GAN (WIP): no conditional generation yet
  - BigGAN (TODO)

## Usage

Train DCGAN using WGAN loss on CelebA, same hyperparameters as paper

```bash
python gan_celeba.py --log_name celeba --accelerator gpu --devices 1 --method wgan --train_g_interval 5 --max_steps 200000 --optimizer RMSprop --lr 5e-5 --batch_size 64
```

Using WGAN-GP loss

```bash
python gan_celeba.py --log_name celeba --accelerator gpu --devices 1 --method wgan-gp --train_g_interval 5 --max_steps 200000 --optimizer Adam --lr 1e-4 --batch_size 64 --beta1 0 --beta2 0.9
```

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
- WGAN and WGAN-GP are not always better than the original GAN loss.
- WGAN: it is necessary to train Discriminator more than Generator (so that Discriminator is a good EMD estimator given a fixed Generator), otherwise Discriminator may collapse `D(x) = D(G(z))`.
- Optimizer: most GANs use Adam with low `beta1` (0.5 or 0.0). Some GANs (WGAN-GP, NVIDIA GANs) require `beta1=0` for stable training. WGAN uses RMSprop. I haven't experimented with other optimizers. SGD probably won't be able to optimize the minimax game. GANs also don't use weight decay.
- Provide label information helps with GAN training. I didn't try modifying Discriminator to classify all classes + fake (suggested by [(Salimans, 2016)](https://proceedings.neurips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)), but Conditional GAN seems to speed up convergence. Conditional GAN probably prevents mode collapse also.
- Progressive GAN: I don't implement progressive growing. Training at 64x64, using most of the training details from the paper, Discriminator outputs explode without Discriminator output L2 penalty (section A.1). The paper also used EMA on Generator, but I haven't tried.
- StyleGAN: mini-batch standard deviation in Discriminator and beta1=0 seem to be important. Tanh is not used in Generator (to force values in [-1,1])
- EMA of the Generator is extremely beneficial. Training DCGAN on CelebA, EMA reduces strange artifacts, makes the generated images smoother and more coherent. [Yazıc 2019](https://arxiv.org/abs/1806.04498) studies this effect. NVIDIA GANs (Progressive GAN, StyleGAN series) and BigGAN use EMA.
