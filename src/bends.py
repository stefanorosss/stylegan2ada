import numpy as np
from functools import partial
import torch as th
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import kornia as K
import kornia.augmentation as kA
import kornia.geometry.transform as kT
import torchvision
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class PackImageGrid(th.nn.Module):

    def __init__(self):
        super(PackImageGrid, self).__init__()

    def pack_image_grid(self, z, chop=2):
        z = z.repeat(chop ** 2, 1, 1, 1)
        n, c, h, w = z.shape

        assert (n == chop * chop)
        z = make_grid(z, nrow=chop, padding=0, range=(0, 1))

        return z.view((1, c, h * chop, w * chop))

    def forward(self, x):
        return self.pack_image_grid(x)


class Tile(th.nn.Module):

    def __init__(self):
        super(Tile, self).__init__()

    def tile(self, x):
        x_high = th.cat((x, K.hflip(x)), axis=3)
        x_low = K.vflip(x_high)
        x = th.cat((x_high, x_low), axis=2)
        return x

    def forward(self, x):
        return self.tile(x)


class Rotate(th.nn.Module):

    def __init__(self, modulation):
        super(Rotate, self).__init__()
        self.modulation = modulation

    def forward(self, x):
        # worst case rotation brings sqrt(2) * max_side_length out-of-frame pixels into frame
        # padding should cover that exactly
        h, w = (x.shape[-2], x.shape[-1])
        padding = int(max(h, w) * (1 - np.sqrt(2) / 2))
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Rotate(b), kA.CenterCrop((h, w)))
        fn = sequential_fn(self.modulation)
        return fn(x)


class Translate(th.nn.Module):

    def __init__(self, modulation):
        super(Translate, self).__init__()
        self.modulation = modulation

    def forward(self, x):
        h, w = (x.shape[-2], x.shape[-1])
        sequential_fn = lambda b: th.nn.Sequential(
            kT.Translate(b),
            kA.CenterCrop((h, w)),
        )
        fn = sequential_fn(self.modulation*h)
        return fn(x)


class Roll(th.nn.Module):

    def __init__(self, modulation):
        super(Roll, self).__init__()
        self.modulation = modulation

    def forward(self, x):
        h, w = (x.shape[-2], x.shape[-1])
        x = th.roll(x, int(-h * self.modulation), -2)
        # x = th.Tensor(gaussian_filter(x.cpu(), 2)).to(x.device)
        return x
