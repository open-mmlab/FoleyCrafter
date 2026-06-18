import json
import os

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(config_path):
    config = json.loads(open(config_path).read())
    config = AttrDict(config)
    return config


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = torch.nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = torch.nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )  # change: 80 --> 512
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            if (k - u) % 2 == 0:
                self.ups.append(
                    weight_norm(
                        ConvTranspose1d(
                            h.upsample_initial_channel // (2**i),
                            h.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        )
                    )
                )
            else:
                self.ups.append(
                    weight_norm(
                        ConvTranspose1d(
                            h.upsample_initial_channel // (2**i),
                            h.upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2 + 1,
                            output_padding=1,
                        )
                    )
                )

            # self.ups.append(weight_norm(
            #     ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
            #                     k, u, padding=(k-u)//2)))

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def dtype(self):
        return self.type

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "vocoder.pt")

        config = get_config(config_path)
        vocoder = cls(config)

        state_dict_g = torch.load(ckpt_path)
        vocoder.load_state_dict(state_dict_g["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder

    @torch.no_grad()
    def inference(self, mels, lengths=None):
        self.eval()
        with torch.no_grad():
            wavs = self(mels).squeeze(1)

        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")

        if lengths is not None:
            wavs = wavs[:, :lengths]

        return wavs
