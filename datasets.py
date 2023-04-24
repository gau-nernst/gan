import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


# https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
# https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/main/serialize.py
class TorchStringArray:
    def __init__(self, list_of_str: list[str]):
        data = [np.frombuffer(x.encode("utf-8"), dtype=np.uint8) for x in list_of_str]
        self.data = np.concatenate(data)
        self.index = np.array([0] + [len(x) for x in data])
        self.index.cumsum(out=self.index)

        self.data = torch.from_numpy(self.data)
        self.index = torch.from_numpy(self.index)

    def __getitem__(self, idx: int) -> str:
        return str(self.data[self.index[idx] : self.index[idx + 1]].numpy(), "utf-8")

    def __len__(self) -> int:
        return self.index.shape[0] - 1


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        files = os.listdir(self.data_dir)
        files.sort()
        self.files = TorchStringArray(files)

    def __getitem__(self, idx: int) -> Tensor:
        return read_image(os.path.join(self.data_dir, self.files[idx]), mode=ImageReadMode.RGB) / 127.5 - 1

    def __len__(self) -> int:
        return len(self.files)
