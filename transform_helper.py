import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps


class PadWhite(object):
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, tuple):
            self.height, self.width = size
        elif isinstance(size, int):
            self.height = self.width = size

    def __call__(self, img):
        if img.size[0] > self.width or img.size[1] > self.height:
            img.thumbnail((self.width, self.height))
        delta_width = self.width - img.size[0]
        delta_height = self.height - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width -
                   pad_width, delta_height-pad_height)
        return ImageOps.expand(img, padding, fill=255)
