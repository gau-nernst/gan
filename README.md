# Generative Adversarial Networks (GANs)

Features:

- Original GAN training
- WGAN and WGAN-GP
- Conditional GAN
- DCGAN
- StyleGAN (WIP)

## Usage

Train DCGAN using WGAN loss on CelebA, same hyperparameters as paper

```bash
python gan_celeba.py --log_name celeba --accelerator gpu --devices 1 --method wgan --train_g_interval 5 --max_steps 200000 --optimizer RMSprop --lr 5e-5 --batch_size 64
```

## Lessons

Some lessons I have learned from implementing and training GANs:

- I could never get MLP-based GANs working. CNN-based GANs are much easier to train.
- Use small batch size e.g. 32 (at least for original GAN. DCGAN paper used 128, WGAN paper used 64). Large batch size will make Discriminator much stronger than Generator. I haven't experimented batch size effect for WGAN, although theoretically WGAN should be able to be trained with large batch (since WGAN wants the best Discriminator given a Generator to estimate the correct Earth mover's distance).
- For transposed convolution, use `kernel_size=4` to avoid checkerboard artifacts.
- DCGAN's Discriminator does not use pooling at the output, but flatten and matmul to 1 output. This is implemented as convolution with kernel size = feature map size. The paper states it helps with convergence speed. For the Generator, DCGAN applies matmul to latent vector and reshape to (512,4,4). This is implemented as convolution transpose with kernel size = output feature map size i.e. 4. All kernel sizes are 4 in Discriminator and Generator.
- Batch norm helps GANs converge much faster. Regarding which mode (training vs evaluation) Batch norm layer should be at different stages, it seems most implementations leave it in training mode during training.
  - For Discriminator, since it is only used in training, a reasonable choice is for Batch norm to always be in training mode (use current batch statistics).
  - For Generator, Batch norm should be in training mode during training, even when it generates samples for training Discriminator. During inference, if the batch size is large, Batch norm can be in training mode also. Generating single image (batch size 1) might be problematic.
  - I haven't explored other norm layers e.g. Layer norm, Instance norm, Adaptive Instance norm.
- To enforce 1-L continuity for Discriminator in WGAN, weight clipping may make Discriminator collapse `D(x) = D(G(z))`. Gradient penalty (WGAN-GP) is not working for me.
- GAN training depends on random seed. Sometimes I can get better (or worse) results by repeating the same training.
- Optimizers: Early GANs use Adam with `beta1=0.5`. WGAN uses RMSprop. I haven't experimented with other optimizers. SGD probably won't be able to optimize the minimax game. GANs also don't use weight decay.
- Provide label information helps with GAN training. I didn't try modifying Discriminator to classify all classes + fake (suggested by [(Salimans, 2016)](https://proceedings.neurips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html)), but Conditional GAN seems to speed up convergence. Conditional GAN probably prevents mode collapse also.
- StyleGAN: mini-batch standard deviation in Discriminator and beta1=0 seem to be important. Tanh is not used in Generator (to force values in [-1,1])
