import argparse
import glob
import os
import os.path as osp
from pathlib import Path

import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy


vision_transform_list = [
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
video_transform = torchvision.transforms.Compose(vision_transform_list)


def args_parse():
    config = argparse.ArgumentParser()
    config.add_argument("--prompt", type=str, default="", help="prompt for audio generation")
    config.add_argument("--nprompt", type=str, default="", help="negative prompt for audio generation")
    config.add_argument("--seed", type=int, default=42, help="ramdom seed")
    config.add_argument("--semantic_scale", type=float, default=1.0, help="visual content scale")
    config.add_argument("--temporal_scale", type=float, default=0.2, help="temporal align scale")
    config.add_argument("--input", type=str, default="examples/sora", help="input video folder path")
    config.add_argument("--ckpt", type=str, default="checkpoints/", help="checkpoints folder path")
    config.add_argument("--save_dir", type=str, default="output/", help="generation result save path")
    config.add_argument(
        "--pretrain",
        type=str,
        default="auffusion/auffusion-full-no-adapter",
        help="audio generator pretrained checkpoint path",
    )
    config.add_argument("--device", type=str, default="cuda")
    config = config.parse_args()
    return config


def build_models(config):
    # download ckpt
    pretrained_model_name_or_path = config.pretrain
    if not os.path.isdir(pretrained_model_name_or_path):
        pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)

    fc_ckpt = "ymzhang319/FoleyCrafter"
    if not os.path.isdir(fc_ckpt):
        fc_ckpt = snapshot_download(fc_ckpt, local_dir=config.ckpt)

    # ckpt path
    temporal_ckpt_path = osp.join(config.ckpt, "temporal_adapter.ckpt")

    # load vocoder
    vocoder_config_path = fc_ckpt
    vocoder = Generator.from_pretrained(vocoder_config_path, subfolder="vocoder").to(config.device)

    # load time_detector
    time_detector_ckpt = osp.join(osp.join(config.ckpt, "timestamp_detector.pth.tar"))
    time_detector = VideoOnsetNet(False)
    time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=config.device, strict=True)

    # load adapters
    pipe = build_foleycrafter().to(config.device)
    ckpt = torch.load(temporal_ckpt_path)

    # load temporal adapter
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    load_gligen_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            load_gligen_ckpt[key[len("module.") :]] = value
        else:
            load_gligen_ckpt[key] = value
    m, u = pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
    print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

    # load semantic adapter
    pipe.load_ip_adapter(
        osp.join(config.ckpt, "semantic"), subfolder="", weight_name="semantic_adapter.bin", image_encoder_folder=None
    )
    ip_adapter_weight = config.semantic_scale
    pipe.set_ip_adapter_scale(ip_adapter_weight)

    return pipe, vocoder, time_detector


def run_inference(config, pipe, vocoder, time_detector):
    controlnet_conditioning_scale = config.temporal_scale
    os.makedirs(config.save_dir, exist_ok=True)

    input_list = glob.glob(f"{config.input}/*.mp4")
    assert len(input_list) != 0, "input list is empty!"

    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed)
    image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder"
    ).to(config.device)
    input_list.sort()
    with torch.no_grad():
        for input_video in input_list:
            print(f" >>> Begin Inference: {input_video} <<< ")
            frames, duration = read_frames_with_moviepy(input_video, max_frame_nums=150)

            time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
            time_frames = video_transform(time_frames)
            time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
            preds = time_detector(time_frames)
            preds = torch.sigmoid(preds)

            # duration
            # import ipdb; ipdb.set_trace()
            time_condition = [
                -1 if preds[0][int(i / (1024 / 10 * duration) * 150)] < 0.5 else 1
                for i in range(int(1024 / 10 * duration))
            ]
            time_condition = time_condition + [-1] * (1024 - len(time_condition))
            # w -> b c h w
            time_condition = (
                torch.FloatTensor(time_condition)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 1, 256, 1)
                .to("cuda")
            )
            images = image_processor(images=frames, return_tensors="pt").to("cuda")
            image_embeddings = image_encoder(**images).image_embeds
            image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
            neg_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)

            name = Path(input_video).stem
            name = name.replace("+", " ")

            sample = pipe(
                prompt=config.prompt,
                negative_prompt=config.nprompt,
                ip_adapter_image_embeds=image_embeddings,
                image=time_condition,
                # audio_length_in_s=10,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=25,
                height=256,
                width=1024,
                output_type="pt",
                generator=generator,
                # guidance_scale=0,
            )
            audio_img = sample.images[0]
            audio = denormalize_spectrogram(audio_img)
            audio = vocoder.inference(audio, lengths=160000)[0]
            audio_save_path = osp.join(config.save_dir, "audio")
            video_save_path = osp.join(config.save_dir, "video")
            os.makedirs(audio_save_path, exist_ok=True)
            os.makedirs(video_save_path, exist_ok=True)
            audio = audio[: int(duration * 16000)]

            save_path = osp.join(audio_save_path, f"{name}.wav")
            sf.write(save_path, audio, 16000)

            audio = AudioFileClip(osp.join(audio_save_path, f"{name}.wav"))
            video = VideoFileClip(input_video)
            audio = audio.subclip(0, duration)
            video.audio = audio
            video = video.subclip(0, duration)
            os.makedirs(video_save_path, exist_ok=True)
            video.write_videofile(osp.join(video_save_path, f"{name}.mp4"))


if __name__ == "__main__":
    config = args_parse()
    pipe, vocoder, time_detector = build_models(config)
    run_inference(config, pipe, vocoder, time_detector)
