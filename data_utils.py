import numpy as np
import torch


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
        return str(memoryview(self.data[self.index[idx] : self.index[idx + 1]].numpy()), "utf-8")

    def __len__(self) -> int:
        return self.index.shape[0] - 1
