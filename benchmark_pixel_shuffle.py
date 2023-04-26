import time

import torch
import torch.nn.functional as F


torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


device = "cuda"
dtype = torch.half
do_compile = False
N, C, H, W = 4, 64, 256, 256
imgs = torch.randn(N, C, H, W, device=device, dtype=dtype)
kernel1 = torch.randn(C * 4, C, 3, 3, device=device, dtype=dtype)
kernel2 = kernel1.flip(2, 3).reshape(C, 2, 2, C, 3, 3).permute(3, 0, 4, 1, 5, 2).reshape(C, C, 6, 6)


def pixel_shuffle(imgs, kernel):
    return F.pixel_shuffle(F.conv2d(imgs, kernel, padding=1), 2)


def conv_transpose(imgs, kernel):
    return F.conv_transpose2d(imgs, kernel, stride=2, padding=2)


if do_compile:
    pixel_shuffle = torch.compile(pixel_shuffle)
    conv_transpose = torch.compile(conv_transpose)

out1 = pixel_shuffle(imgs, kernel1)
print(out1.shape)

out2 = conv_transpose(imgs, kernel2)
print(out2.shape)

error = (out2 - out1).abs()
print("mean diff:", error.mean())
print("max diff:", error.max())


def cuda_sync():
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark(fn, inputs):
    out = fn(*inputs)

    cuda_sync()
    time0 = time.perf_counter()
    for _ in range(100):
        out = fn(*inputs)
        cuda_sync()
    return 100 / (time.perf_counter() - time0)


print(f"Pixel shuffle: {benchmark(pixel_shuffle, (imgs, kernel1)):.4f} it/s")
print(f"Conv transpose: {benchmark(conv_transpose, (imgs, kernel2)):.4f} it/s")
