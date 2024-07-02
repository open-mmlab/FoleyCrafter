import torch.nn as nn

from ..onset import VideoOnsetNet


class TimeDetector(nn.Module):
    def __init__(self, video_length=150, audio_length=1024):
        super(TimeDetector, self).__init__()
        self.pred_net = VideoOnsetNet(pretrained=False)
        self.soft_fn = nn.Tanh()
        self.up_sampler = nn.Linear(video_length, audio_length)

    def forward(self, inputs):
        x = self.pred_net(inputs)
        x = self.up_sampler(x)
        x = self.soft_fn(x)
        return x
