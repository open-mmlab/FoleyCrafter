<p align="center">
<img src='assets/foleycrafter.png' style="text-align: center; width: 134px" >
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Project Page](https://img.shields.io/badge/FoleyCrafter-Website-green)](https://foleycrafter.github.io)
<a target="_blank" href="https://huggingface.co/spaces/ymzhang319/FoleyCrafter">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ymzhang319/FoleyCrafter)

</div>

# FoleyCrafter

Sound effects are the unsung heroes of cinema and gaming, enhancing realism, impact, and emotional depth for an immersive audiovisual experience. **FoleyCrafter** is a video-to-audio generation framework which can produce realistic sound effects semantically relevant and synchronized with videos.

**Your star is our fuel! <img alt="" width="30" src="https://camo.githubusercontent.com/2f4f0d02cdf79dc1ff8d2b053b4410b13bc2e39cbc8a96fcdc6f06538a3d6d2b/68747470733a2f2f656d2d636f6e74656e742e7a6f626a2e6e65742f736f757263652f616e696d617465642d6e6f746f2d636f6c6f722d656d6f6a692f3335362f736d696c696e672d666163652d776974682d6865617274735f31663937302e676966"> We're revving up the engines with it! <img alt="" width="30" src="https://camo.githubusercontent.com/028a75f875b8c3aa1b3c80bbf7dd27973c4bb654fffcf0bdc0b6f1b0674ce481/68747470733a2f2f656d2d636f6e74656e742e7a6f626a2e6e65742f736f757263652f74656c656772616d2f3338362f737061726b6c65735f323732382e77656270">**


[FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds]()

[Yiming Zhang](https://github.com/ymzhang0319),
[Yicheng Gu](https://github.com/VocodexElysium),
[Yanhong Zeng†](https://zengyh1900.github.io/),
[Zhening Xing](https://github.com/LeoXing1996/),
[Yuancheng Wang](https://github.com/HeCheng0625),
[Zhizheng Wu](https://drwuz.com/),
[Kai Chen†](https://chenkai.site/)

(†Corresponding Author)


## What's New
- [ ] A more powerful one :stuck_out_tongue_closed_eyes: .
- [ ] Release training code.
- [x] `2024/07/01` Release the model and code of FoleyCrafter.

## Setup

### Prepare Environment
Use the following command to install dependencies:
```bash
# install conda environment
conda env create -f requirements/environment.yaml
conda activate foleycrafter

# install GIT LFS for checkpoints download
conda install git-lfs
git lfs install
```

### Download Checkpoints
The checkpoints will be downloaded automatically by running `inference.py`.

You can also download manually using following commands.
<li> Download the text-to-audio base model. We use Auffusion</li>

```bash
git clone https://huggingface.co/auffusion/auffusion-full-no-adapter checkpoints/auffusion
```

<li> Download FoleyCrafter</li>

```bash
git clone https://huggingface.co/ymzhang319/FoleyCrafter checkpoints/
```

Put checkpoints as follows:
```
└── checkpoints
    ├── semantic
    │   ├── semantic_adapter.bin
    ├── vocoder
    │   ├── vocoder.pt
    │   ├── config.json
    ├── temporal_adapter.ckpt
    │   │
    └── timestamp_detector.pth.tar
```

## Gradio demo

You can launch the Gradio interface for FoleyCrafter by running the following command:

```bash
python app.py
```



## Inference
### Video To Audio Generation
```bash
python inference.py --save_dir=output/sora/
```

Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Input Video</p></td>
  <td><p style="text-align: center">Generated Audio</p></td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309262-d7c89984-c567-4ca7-8e2d-8f49d84bda4a.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122032Z&X-Amz-Expires=300&X-Amz-Signature=5b13f216056dedca2705233038dbb22f73023d2c1deaf3b03972d7b91c1bbab5&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

 </td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309725-0dfa72a2-1466-46e6-9611-3e1cbff707fe.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122123Z&X-Amz-Expires=300&X-Amz-Signature=314648ed216620b2d926395d34602c70da500eb9e865e839de6907ed1b0d0bd1&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309166-16206bb8-9c5e-4e9d-9d73-bc251e5658fd.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122143Z&X-Amz-Expires=300&X-Amz-Signature=43c3e2c687846eb3ba118237628b747a78c403ea4f21739fe2d423724f7b426c&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309768-90c42af6-0d24-4a05-98d4-64e23467c4bb.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122213Z&X-Amz-Expires=300&X-Amz-Signature=cfed43cd2710bf73b84b6c3ebe8debd1e0b098bdc24a1a14f6531499d01c278e&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309601-e711b7c5-1614-4d39-8b1e-c54e28eec809.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122221Z&X-Amz-Expires=300&X-Amz-Signature=4c4680eb6c541433e4505fb2b5f5a5cc8d3e5708d9f2675a98cbb556cd5d59f5&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309802-2db7f130-0c25-45c2-ad4d-bf86c5468b1f.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122243Z&X-Amz-Expires=300&X-Amz-Signature=2c32069318f60f03ee9a3185a7e2833c534ea601a02a5ff025b46c2abbc5b120&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309637-6c2f106d-6b98-41ac-80ba-734636321f8c.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122305Z&X-Amz-Expires=300&X-Amz-Signature=04adb2cd80785a245ce704837ba9932d3646d375c51c53e7b70e6861ec7f6b4a&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342309836-77391524-9b31-4602-ad42-0876e0c16794.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122317Z&X-Amz-Expires=300&X-Amz-Signature=3020c12591592a106efcb1aaa22237093737afaa55ede3880d3cbf9cd80b7482&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
</table>

- Temporal Alignment with Visual Cues
```bash
python inference.py \
--temporal_align \
--input=input/avsync \
--save_dir=output/avsync/
```

Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Ground Truth</p></td>
  <td><p style="text-align: center">Generated Audio</p></td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310778-bcc0f16d-6d1b-468d-a775-81b8f2d98ea6.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122327Z&X-Amz-Expires=300&X-Amz-Signature=205b8e190a428b3ddee41fe2549080b4f50fd8bb10ef78d650fc05add85ccbab&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310418-8433e05c-8600-4cd6-8a68-ead536159204.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122337Z&X-Amz-Expires=300&X-Amz-Signature=3fc37a305511c1c8b7bdfc9b9b5bd0485fd584400af087939a4c08218ab33538&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310801-3d6fd80d-de6b-4815-ac6a-f81772709e4c.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122349Z&X-Amz-Expires=300&X-Amz-Signature=fc31f139a1f9c7606657fa457f1271a8a44cb39a13454939c030ccdafe2d3068&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310491-dfaf41e7-487e-47ff-8e8a-fe7cb4fb1942.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122356Z&X-Amz-Expires=300&X-Amz-Signature=6353a935194a08bc081fa873e3c6582fb175874d3112f8f8f96614a5e542ef03&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310825-6834f00f-95e8-4a2c-b864-b4fe57801836.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122406Z&X-Amz-Expires=300&X-Amz-Signature=c67b2f113b0db790a8495d1a4ab4c0d230db5f53a9062a497bcca3e57f9600aa&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342310543-5a2c363b-623c-4329-be0e-a151e5bb56a6.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122415Z&X-Amz-Expires=300&X-Amz-Signature=d8b0bfc28716e0e03694b3e590aca29450fa7788aa39e748130ee12a15d614e9&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
</table>

### Text-based Video to Audio Generation

- Using Prompt

```bash
# case1
python inference.py \
--input=input/PromptControl/case1/ \
--seed=10201304011203481429 \
--save_dir=output/PromptControl/case1/

python inference.py \
--input=input/PromptControl/case1/ \
--seed=10201304011203481429 \
--prompt='noisy, people talking' \
--save_dir=output/PromptControl/case1_prompt/

# case2
python inference.py \
--input=input/PromptControl/case2/ \
--seed=10021049243103289113 \
--save_dir=output/PromptControl/case2/

python inference.py \
--input=input/PromptControl/case2/ \
--seed=10021049243103289113 \
--prompt='seagulls' \
--save_dir=output/PromptControl/case2_prompt/
```
Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Generated Audio</p></td>
  <td><p style="text-align: center">Generated Audio</p></td>
<tr>
<tr>
  <td><p style="text-align: center">Without Prompt</p></td>
  <td><p style="text-align: center">Prompt: <b>noisy, people talking</b></p></td>
<tr>
<tr>
  <td>


https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311425-8dd543cb-0df2-441e-b6d0-86048dbeb73d.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122426Z&X-Amz-Expires=300&X-Amz-Signature=b872b8eaf51a5022aee1daf0283d92e53a70e109c6b9f1e6a4da238a3708ea45&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188


</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311493-62a08024-581c-4716-a030-aef194beddc5.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122439Z&X-Amz-Expires=300&X-Amz-Signature=647eb0dc32bf7c0d739ccbe875826b1a67f54e0dc84e0be70e0e128ae2fdb73d&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td><p style="text-align: center">Without Prompt</p></td>
  <td><p style="text-align: center">Prompt: <b>seagulls</b></p></td>
<tr>
<tr>
  <td>



https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311538-1f81f91e-efc0-41ed-bdcb-c5c6ff976c5b.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122447Z&X-Amz-Expires=300&X-Amz-Signature=d19761788775893e77e42b9312b4c26cb85aedd7c6dc249eaf68ff1f650e1942&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188



</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311595-695668ed-46a1-47b2-b5fd-3aa4286d695e.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122500Z&X-Amz-Expires=300&X-Amz-Signature=bf7d69ab8c74154ee8ac5682f64ce29a310ea2d0365f620893a45899d62a3f80&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
</table>

- Using Negative Prompt
```bash
# case 3
python inference.py \
--input=input/PromptControl/case3/ \
--seed=10041042941301238011 \
--save_dir=output/PromptControl/case3/

python inference.py \
--input=input/PromptControl/case3/ \
--seed=10041042941301238011 \
--nprompt='river flows' \
--save_dir=output/PromptControl/case3_nprompt/

# case4
python inference.py \
--input=input/PromptControl/case4/ \
--seed=10014024412012338096 \
--save_dir=output/PromptControl/case4/

python inference.py \
--input=input/PromptControl/case4/ \
--seed=10014024412012338096 \
--nprompt='noisy, wind noise' \
--save_dir=output/PromptControl/case4_nprompt/

```
Results:
<table class='center'>
<tr>
  <td><p style="text-align: center">Generated Audio</p></td>
  <td><p style="text-align: center">Generated Audio</p></td>
<tr>
<tr>
  <td><p style="text-align: center">Without Prompt</p></td>
  <td><p style="text-align: center">Negative Prompt: <b>river flows</b></p></td>
<tr>
<tr>
  <td>



https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311656-cdc69cf1-88f8-4861-b888-bdb82358b9c5.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122517Z&X-Amz-Expires=300&X-Amz-Signature=1731e655b2f0bb7f4a7af737ee065a01f98fccb1c54ef48ae775ad65ec67eda5&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188



</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311702-cd259522-84f4-44cb-862f-c4dcfb57e5c4.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122527Z&X-Amz-Expires=300&X-Amz-Signature=d846ac60df25ff18de1861daeff380b7bc8ca21c04e7dce139ed44abbf9aaa22&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
<tr>
  <td><p style="text-align: center">Without Prompt</p></td>
  <td><p style="text-align: center">Negative Prompt: <b>noisy, wind noise</b></p></td>
<tr>
<tr>
  <td>


https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311785-5ca9c050-a928-4dc2-b620-d843a3ae72f5.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122533Z&X-Amz-Expires=300&X-Amz-Signature=151fe4da521f9f48ff245ef5bd7c6964f1dfc652be0fe8de4c151ba59e87d2d6&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188


</td>
  <td>

https://github-production-user-asset-6210df.s3.amazonaws.com/134203169/342311844-28d6abe3-d5a8-4a7f-9f4d-3cc8411affba.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240624T122544Z&X-Amz-Expires=300&X-Amz-Signature=39a259ca76a12a57d47ad74c8cf92af12af9e3db34084f5f82c79b6a62356e9a&X-Amz-SignedHeaders=host&actor_id=134203169&key_id=0&repo_id=812946188

</td>
<tr>
</table>

### Commandline Usage Parameters
```console
options:
  -h, --help            show this help message and exit
  --prompt PROMPT       prompt for audio generation
  --nprompt NPROMPT     negative prompt for audio generation
  --seed SEED           ramdom seed
  --temporal_align TEMPORAL_ALIGN
                        use temporal adapter or not
  --temporal_scale TEMPORAL_SCALE
                        temporal align scale
  --semantic_scale SEMANTIC_SCALE
                        visual content scale
  --input INPUT         input video folder path
  --ckpt CKPT           checkpoints folder path
  --save_dir SAVE_DIR   generation result save path
  --pretrain PRETRAIN   generator checkpoint path
  --device DEVICE
```


<!-- ## Train
To train FoleyCrafter, run the training script of semantic/temporal adapter individually.

```bash
# semantic adapter
python train/train_semantic_adapter.py \
--config=configs/train/train_semantic_adapter.yaml

# temporal adapter
python train/train_temporal_adapter.py \
--config=configs/train/train_temporal_adapter.yaml
``` -->

## Contact Us

**Yiming Zhang**: zhangyiming@pjlab.org.cn

**YiCheng Gu**: yichenggu@link.cuhk.edu.cn

**Yanhong Zeng**: zengyanhong@pjlab.org.cn

## LICENSE
Please check [Apache-2.0 license](./LICENSE.txt) for details.

## Acknowledgements
The code is built upon [Auffusion](https://github.com/happylittlecat2333/Auffusion), [CondFoleyGen](https://github.com/XYPB/CondFoleyGen) and [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN).

We recommend a professional and user-frienly project [Amphion]()
