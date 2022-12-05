# Generative Adversarial Networks (GANs)

Some lessons I have learned from implementing and training GANs:

- I could never get MLP-based GANs working. CNN-based GANs are much easier to train.
- Use small batch size e.g. 32 (at least for original GAN. DCGAN paper used batch size 128). Large batch size will make Discriminator much stronger than Generator. I haven't experimented batch size effect for WGAN, although theoretically WGAN should be able to be trained with large batch (since WGAN wants the best Discriminator given a Generator to estimate the correct Earth mover's distance).
- For transposed convolution, use `kernel_size=4` to avoid checkerboard artifacts.
- Batch norm helps GANs converge much faster. However, there is no reference about which mode (training vs evaluation) Batch norm layer should be at different stages.
  - For Discriminator, since it is only used in training, a reasonable choice is for Batch norm to always be in training mode (use current batch statistics).
  - For Generator, Batch norm should be in training mode during training, even when it generates samples for training Discriminator. During inference, if the batch size is large, Batch norm can be in training mode also, since the input (latent vectors) follows the same distribution. Batch norm in evaluation mode for generated images may be ill-defined, since the underlying distribution changes as the Generator is optimized, so the running statistics is not accurate.
  - I haven't explored other norm layers e.g. Layer norm, Instance norm, Adaptive Instance norm.
- To enforce 1-L continuity for Discriminator in WGAN, weight clipping may make Discriminator collapse `D(x) = D(G(z))`. Gradient penalty (WGAN-GP) is not working for me.
