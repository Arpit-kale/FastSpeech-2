"""Modified from ming024/FastSpeech2 ``hifigan/models.py`` (originally jik876/hifi-gan, MIT).

Only the V1 ``Generator`` and ``ResBlock1`` are kept (the discriminators are not
needed for inference). Parameter names are unchanged, so the universal
checkpoint shipped with the reference loads with ``strict=True``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
V1_CONFIG = json.loads((Path(__file__).with_name("config.json")).read_text())

LRELU_SLOPE = 0.1


def _padding(k: int, d: int) -> int:
    return (k * d - d) // 2


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=_padding(kernel, d))) for d in dilations]
        )
        self.convs2 = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel, 1, dilation=1, padding=_padding(kernel, 1))) for _ in dilations]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in list(self.convs1) + list(self.convs2):
            remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self, cfg: dict | None = None, n_mels: int = 80):
        cfg = cfg or V1_CONFIG
        super().__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])
        ch = cfg["upsample_initial_channel"]
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, ch, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(nn.ConvTranspose1d(ch // 2**i, ch // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            c = ch // 2 ** (i + 1)
            for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock1(c, k, d))
        self.conv_post = weight_norm(nn.Conv1d(c, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU_SLOPE))
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
