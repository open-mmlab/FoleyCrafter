import os
import os.path as osp
import random
from argparse import ArgumentParser
from datetime import datetime

import gradio as gr
import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from foleycrafter.models.onset import torch_utils
from foleycrafter.models.time_detector.model import VideoOnsetNet
from foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy


os.environ["GRADIO_TEMP_DIR"] = "./tmp"

sample_idx = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="example/config/base.yaml")
parser.add_argument("--server-name", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=11451)
parser.add_argument("--share", action="store_true")

parser.add_argument("--save-path", default="samples")

args = parser.parse_args()


N_PROMPT = ""


class FoleyController:
    def __init__(self):
        # config dirs
        self.basedir = os.getcwd()
        self.model_dir = os.path.join(self.basedir, "models")
        self.savedir = os.path.join(self.basedir, args.save_path, datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = None

        self.loaded = False

        self.load_model()

    def load_model(self):
        gr.Info("Start Load Models...")
        print("Start Load Models...")

        # download ckpt
        pretrained_model_name_or_path = "auffusion/auffusion-full-no-adapter"
        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(
                pretrained_model_name_or_path, local_dir="models/auffusion"
            )

        fc_ckpt = "ymzhang319/FoleyCrafter"
        if not os.path.isdir(fc_ckpt):
            fc_ckpt = snapshot_download(fc_ckpt, local_dir="models/")

        # set model config
        temporal_ckpt_path = osp.join(self.model_dir, "temporal_adapter.ckpt")

        # load vocoder
        vocoder_config_path = "./models/auffusion"
        self.vocoder = Generator.from_pretrained(vocoder_config_path, subfolder="vocoder").to(self.device)

        # load time detector
        time_detector_ckpt = osp.join(osp.join(self.model_dir, "timestamp_detector.pth.tar"))
        time_detector = VideoOnsetNet(False)
        self.time_detector, _ = torch_utils.load_model(
            time_detector_ckpt, time_detector, strict=True, device=self.device
        )

        self.pipeline = build_foleycrafter().to(self.device)
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
        m, u = self.pipeline.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
        print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder"
        ).to(self.device)

        self.pipeline.load_ip_adapter(
            fc_ckpt, subfolder="semantic", weight_name="semantic_adapter.bin", image_encoder_folder=None
        )

        gr.Info("Load Finish!")
        print("Load Finish!")
        self.loaded = True

        return "Load"

    def foley(
        self,
        input_video,
        prompt_textbox,
        negative_prompt_textbox,
        ip_adapter_scale,
        temporal_scale,
        sampler_dropdown,
        sample_step_slider,
        cfg_scale_slider,
        seed_textbox,
    ):
        vision_transform_list = [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.CenterCrop((112, 112)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        video_transform = torchvision.transforms.Compose(vision_transform_list)
        if not self.loaded:
            raise gr.Error("Error with loading model")
        generator = torch.Generator()
        if seed_textbox != "":
            torch.manual_seed(int(seed_textbox))
            generator.manual_seed(int(seed_textbox))
        max_frame_nums = 15
        frames, duration = read_frames_with_moviepy(input_video, max_frame_nums=max_frame_nums)
        if duration >= 10:
            duration = 10
        time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        time_frames = video_transform(time_frames)
        time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
        preds = self.time_detector(time_frames)
        preds = torch.sigmoid(preds)

        # duration
        time_condition = [
            -1 if preds[0][int(i / (1024 / 10 * duration) * max_frame_nums)] < 0.5 else 1
            for i in range(int(1024 / 10 * duration))
        ]
        time_condition = time_condition + [-1] * (1024 - len(time_condition))
        # w -> b c h w
        time_condition = torch.FloatTensor(time_condition).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 256, 1)

        images = self.image_processor(images=frames, return_tensors="pt").to(self.device)
        image_embeddings = self.image_encoder(**images).image_embeds
        image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
        neg_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)
        self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
        sample = self.pipeline(
            prompt=prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            ip_adapter_image_embeds=image_embeddings,
            image=time_condition,
            controlnet_conditioning_scale=float(temporal_scale),
            num_inference_steps=sample_step_slider,
            height=256,
            width=1024,
            output_type="pt",
            generator=generator,
        )
        name = "output"
        audio_img = sample.images[0]
        audio = denormalize_spectrogram(audio_img)
        audio = self.vocoder.inference(audio, lengths=160000)[0]
        audio_save_path = osp.join(self.savedir_sample, "audio")
        os.makedirs(audio_save_path, exist_ok=True)
        audio = audio[: int(duration * 16000)]

        save_path = osp.join(audio_save_path, f"{name}.wav")
        sf.write(save_path, audio, 16000)

        audio = AudioFileClip(osp.join(audio_save_path, f"{name}.wav"))
        video = VideoFileClip(input_video)
        audio = audio.subclip(0, duration)
        video.audio = audio
        video = video.subclip(0, duration)
        video.write_videofile(osp.join(self.savedir_sample, f"{name}.mp4"))
        save_sample_path = os.path.join(self.savedir_sample, f"{name}.mp4")

        return save_sample_path


controller = FoleyController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            '<h1 style="height: 136px; display: flex; align-items: center; justify-content: space-around;"><span style="height: 100%; width:136px;"><img src="assets/foleycrafter.png" alt="logo" style="height: 100%; width:auto; object-fit: contain; margin: 0px 0px; padding: 0px 0px;"></span><strong style="font-size: 40px;">FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds</strong></h1>'
        )
        with gr.Row():
            gr.Markdown(
                "<div align='center'><font size='5'><a href='https://foleycrafter.github.io/'>Project Page</a> &ensp;"  # noqa
                "<a href='https://arxiv.org/abs/xxxx.xxxxx/'>Paper</a> &ensp;"
                "<a href='https://github.com/open-mmlab/foleycrafter'>Code</a> &ensp;"
                "<a href='https://huggingface.co/spaces/ymzhang319/FoleyCrafter'>Demo</a> </font></div>"
            )

        with gr.Column(variant="panel"):
            with gr.Row(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        init_img = gr.Video(label="Input Video")
                    with gr.Row():
                        prompt_textbox = gr.Textbox(value="", label="Prompt", lines=1)
                    with gr.Row():
                        negative_prompt_textbox = gr.Textbox(value=N_PROMPT, label="Negative prompt", lines=1)

                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(
                            label="Sampling method",
                            choices=list(scheduler_dict.keys()),
                            value=list(scheduler_dict.keys())[0],
                        )
                        sample_step_slider = gr.Slider(
                            label="Sampling steps", value=25, minimum=10, maximum=100, step=1
                        )

                    cfg_scale_slider = gr.Slider(label="CFG Scale", value=7.5, minimum=0, maximum=20)
                    ip_adapter_scale = gr.Slider(label="Visual Content Scale", value=1.0, minimum=0, maximum=1)
                    temporal_scale = gr.Slider(label="Temporal Align Scale", value=0.0, minimum=0.0, maximum=1.0)

                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=42)
                        seed_button = gr.Button(value="\U0001f3b2", elem_classes="toolbutton")
                    seed_button.click(fn=lambda x: random.randint(1, 1e8), outputs=[seed_textbox], queue=False)

                    generate_button = gr.Button(value="Generate", variant="primary")

                result_video = gr.Video(label="Generated Audio", interactive=False)

            generate_button.click(
                fn=controller.foley,
                inputs=[
                    init_img,
                    prompt_textbox,
                    negative_prompt_textbox,
                    ip_adapter_scale,
                    temporal_scale,
                    sampler_dropdown,
                    sample_step_slider,
                    cfg_scale_slider,
                    seed_textbox,
                ],
                outputs=[result_video],
            )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(3)
    demo.launch(
        server_name=args.server_name, server_port=args.port, share=args.share, allowed_paths=["./foleycrafter.png"]
    )
