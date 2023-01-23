import contextlib
import itertools
import sys
import time

import torch

from modelling.nvidia_ops import UpFIRDn2d, upfirdn2d
from nvidia_ops.ops.upfirdn2d import upfirdn2d as nvidia_upfirdn2d

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def benchmark(func, inputs, n_backward=0, N=100):
    def _func():
        out = func(*inputs) ** (n_backward + 1)
        for _ in range(n_backward):
            (out,) = torch.autograd.grad(out.sum(), data, create_graph=True)

    for _ in range(10):
        _func()

    torch.cuda.synchronize()
    time0 = time.perf_counter()
    for _ in range(N):
        _func()
        torch.cuda.synchronize()
    return N / (time.perf_counter() - time0)


device = "cuda"
kernel = torch.randn(3, 3, device=device)

with contextlib.redirect_stdout(sys.stderr):
    data = torch.randn(4, 512, 8, 8, device=device, requires_grad=True)
    out1 = upfirdn2d(data, kernel, 1, 1, 1, 1, 1, 1)
    out2 = nvidia_upfirdn2d(data, kernel.flip(0, 1), padding=1)
assert torch.allclose(out1, out2)

N = 100

print("input_size,n_backward,upfirdn2d,upfirdn2d_gradfix,nvidia_upfirdn2d,upfirdn2d_float16,upfirdn2d_gradfix_float16")
input_sizes = [
    (32, 512, 4, 4),  # StyleGAN/StyleGAN2 structure
    (32, 512, 8, 8),
    (32, 512, 16, 16),
    (32, 512, 32, 32),
    (32, 256, 64, 64),
    (32, 128, 128, 128),
    (32, 64, 256, 256),
    (32, 32, 512, 512),
    (32, 16, 1024, 1024),
]
for input_size, n_backward in itertools.product(input_sizes, range(3)):
    print(f"{input_size=}, {n_backward=}", file=sys.stderr)
    data = torch.randn(input_size, device=device, requires_grad=True)
    kernel = torch.randn(3, 3, device=device)

    d1 = benchmark(upfirdn2d, (data, kernel, 1, 1, 1, 1, 1, 1), n_backward, N)
    d2 = benchmark(UpFIRDn2d.apply, (data, kernel, 1, 1, 1, 1, 1, 1), n_backward, N)
    d3 = benchmark(nvidia_upfirdn2d, (data, kernel, 1, 1, 1), n_backward, N)

    data = data.half()
    kernel = kernel.half()
    d4 = benchmark(upfirdn2d, (data, kernel, 1, 1, 1, 1, 1, 1), n_backward, N)
    d5 = benchmark(UpFIRDn2d.apply, (data, kernel, 1, 1, 1, 1, 1, 1), n_backward, N)

    print('"' + str(input_size) + '"', n_backward, d1, d2, d3, d4, d5, sep=",", flush=True)
