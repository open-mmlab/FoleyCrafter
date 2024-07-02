import glob
import io
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s, flush=True)


@torch.no_grad()
def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["window_size"],
        win_length=audio_cfg["window_size"],
        hop_length=audio_cfg["hop_size"],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=64,
        f_min=audio_cfg["fmin"],
        f_max=audio_cfg["fmax"],
    ).to(audio_data.device)
    mel = mel(audio_data)
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel  # (T, n_mels)


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class AudioSetStrong(Dataset):
    # read feature and audio
    def __init__(
        self,
        data_path="data/AudioSetStrong/train/feature",
        video_path="data/AudioSetStrong/train/video",
    ):
        super().__init__()
        self.data_path = data_path
        self.data_list = list(self.data_path)
        self.length = len(self.data_list)
        # get video feature
        self.video_path = video_path
        vision_transform_list = [
            transforms.Resize((128, 128)),
            transforms.CenterCrop((112, 112)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.video_transform = transforms.Compose(vision_transform_list)

    def get_batch(self, idx):
        embeds = self.data_list[idx]
        mel = embeds["mel"]
        save_bsz = mel.shape[0]
        audio_info = embeds["audio_info"]
        text_embeds = embeds["text_embeds"]

        # audio_info['label_list'] = np.array(audio_info['label_list'])
        audio_info_array = np.array(audio_info["label_list"])
        prompts = []
        for i in range(save_bsz):
            prompts.append(", ".join(audio_info_array[i, : audio_info["event_num"][i]].tolist()))

        return mel, audio_info, text_embeds, prompts

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                mel, audio_info, text_embeds, prompts, videos = self.get_batch(idx)
                break
            except Exception:
                zero_rank_print(" >>> load error <<<")
                idx = random.randint(0, self.length - 1)
        sample = {
            "mel": mel,
            "audio_info": audio_info,
            "text_embeds": text_embeds,
            "prompts": prompts,
            "videos": videos,
        }
        return sample


class VGGSound(Dataset):
    # read feature and audio
    def __init__(
        self,
        data_path="data/VGGSound/train/video",
        visual_data_path="data/VGGSound/train/feature",
    ):
        super().__init__()
        self.data_path = data_path
        self.visual_data_path = visual_data_path
        self.embeds_list = glob.glob(f"{self.data_path}/*.pt")
        self.visual_list = glob.glob(f"{self.visual_data_path}/*.pt")
        self.length = len(self.embeds_list)

    def get_batch(self, idx):
        embeds = torch.load(self.embeds_list[idx], map_location="cpu")
        visual_embeds = torch.load(self.visual_list[idx], map_location="cpu")

        # audio_embeds  = embeds['audio_embeds']
        visual_embeds = visual_embeds["visual_embeds"]
        # video_name = embeds["video_name"]
        text = embeds["text"]
        mel = embeds["mel"]

        audio = mel

        return visual_embeds, audio, text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                visual_embeds, audio, text = self.get_batch(idx)
                break
            except Exception:
                zero_rank_print("load error")
                idx = random.randint(0, self.length - 1)
        sample = {"visual_embeds": visual_embeds, "audio": audio, "text": text}
        return sample
