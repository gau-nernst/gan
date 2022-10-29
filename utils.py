from torchvision.utils import make_grid
from torch.utils.tensorboard.writer import SummaryWriter


def repeat_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data


class TensorboardWriter(SummaryWriter):
    def add_images(self, tag, images, *args, nrow=8, **kwargs):
        grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
        self.add_image(tag, grid, *args, **kwargs)
