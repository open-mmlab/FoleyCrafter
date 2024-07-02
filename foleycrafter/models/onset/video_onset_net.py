# Copied from specvqgan/onset_baseline/models/video_onset_net.py

import torch
import torch.nn as nn

from .r2plus1d_18 import r2plus1d18KeepTemp


class VideoOnsetNet(nn.Module):
    # Video Onset detection network
    def __init__(self, pretrained):
        super(VideoOnsetNet, self).__init__()
        self.net = r2plus1d18KeepTemp(pretrained=pretrained)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True), nn.Linear(128, 1))

    def forward(self, inputs, loss=False, evaluate=False):
        # import pdb; pdb.set_trace()
        x = inputs["frames"]
        x = self.net(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        x = x.squeeze(-1)

        return x


if __name__ == "__main__":
    model = VideoOnsetNet(False).cuda()
    rand_input = torch.randn((1, 3, 30, 112, 112)).cuda()
    inputs = {"frames": rand_input}
    out = model(inputs)
