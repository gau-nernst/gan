from typing import Callable

import torch
from pytorch_fid.inception import InceptionV3
from torch import Tensor, nn


# there are 2 ways to improve this
# 1. online algorithm for mean and cov, so that we don't need to keep full matrix 10,000 x 2048 in memory
# 2. calculate FID smartly by not
class FID(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.inception = InceptionV3().to(device).eval()
        self.device = device

    def compute_stats(self, closure: Callable[[], Tensor]) -> dict[str, Tensor]:
        all_outputs = torch.empty(10_000, 2048, device=self.device)
        i = 0

        while i < 10_000:
            with torch.no_grad():
                out = self.inception(closure())[0].flatten(1)
            j = min(i + out.shape[0], 10_000)
            all_outputs[i:j] = out[: j - i]
            i = j

        return dict(mean=all_outputs.mean(0).cpu(), cov=all_outputs.T.cov().cpu())

    # https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
    @staticmethod
    def fid_score(stats1: dict[str, Tensor], stats2: dict[str, Tensor]) -> float:
        a = (stats1["mean"] - stats2["mean"]).square().sum(dim=-1)
        b = stats1["cov"].trace() + stats2["cov"].trace()
        c = torch.linalg.eigvals(stats1["cov"] @ stats2["cov"]).sqrt().real.sum(dim=-1)
        return a + b - 2 * c
