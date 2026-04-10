# from visualizer import get_local
# get_local.activate()

import os
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

# Download models
# snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_dir="models/Wan-AI/Wan2.1-I2V-14B-480P")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ],
    torch_dtype=torch.float32,  # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(
    model_manager, torch_dtype=torch.bfloat16, device="cuda"
)
pipe.enable_vram_management(
    num_persistent_param_in_dit=6 * 10**9
)  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.

# Download example image
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=f"data/examples/wan/input_image.jpg"
# )
# image = Image.open("/gpfs/share/home/2201111701/Kaiwencheng/DiffSynth-Studio/examples/wanvideo/data/examples/wan/01.jpg")

selected_data = {
    "DAVIS_25": ["train_000009", "test_000019", "train_000067", "train_000069"],
    "RealEs_25": [x.rjust(6, "0") for x in ["34", "59", "62", "95", "96", "132"]],
    "human_hq_25": [
        x.rjust(6, "0") for x in ["29", "106", "145", "202", "227", "230", "244", "265"]
    ],
}

names = []
images = []
prompts = []
save_dirs = []

base_image_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/data/examples/gt"
base_save_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/results/examples/wan_all/auto_generated"


def list_one_level_subdirs(root_dir):
    return [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]


def extract_video_description(md_file_path):
    with open(md_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    description_lines = []
    in_description = False

    for line in lines:
        if line.strip().startswith("# 视频生成描述"):
            in_description = True
            continue
        if in_description and line.strip().startswith("## "):
            break
        if in_description:
            description_lines.append(line.strip())

    return " ".join(description_lines)


for dataset_name in selected_data.keys():
    dataset_image_dir = os.path.join(base_image_dir, f"{dataset_name}_gt")
    dataset_save_dir = os.path.join(base_save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    indices = list_one_level_subdirs(dataset_image_dir)
    indices.sort()

    for i in indices:
        image_dir = os.path.join(dataset_image_dir, i)
        names.append(f"{dataset_name}-{i}")

        images.append(
            [
                Image.open(os.path.join(image_dir, "image00.png")).convert("RGB"),
                Image.open(os.path.join(image_dir, "image24.png")).convert("RGB"),
            ]
        )

        prompts.append(
            extract_video_description(os.path.join(image_dir, "video_metadata_old.md"))
        )

        save_dirs.append(os.path.join(dataset_save_dir, f"{os.path.basename(i)}.mp4"))

valid_names = []
valid_images = []
valid_prompts = []
valid_save_dirs = []

for name, imgs, prompt, dir in zip(names, images, prompts, save_dirs):
    if not os.path.exists(dir):
        valid_names.append(name)
        valid_images.append(imgs)
        valid_prompts.append(prompt)
        valid_save_dirs.append(dir)

# Image-to-video
for i, (name, imgs, prompt, dir) in enumerate(zip(valid_names, valid_images, valid_prompts, valid_save_dirs)):
    if i > 30:
        print(f"Start-End-Frame generating {name}...")
        print(f"Detected {len(imgs)} input images.")
        assert len(imgs) in {1, 2, 3}, "Only support 1 or 2 or 3 input key frame(s)."
        video = pipe(
            prompt=prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            num_frames=82,
            input_images=imgs,
            middle_frame_mask=None,
            middle_frame_position=10,
            num_inference_steps=50,
            seed=0,
            tiled=False,
        )
        save_video(video, dir, fps=15, quality=5)

        # cache = get_local.cache

        # import numpy as np

        # np.save(f"/gpfs/share/home/2201111701/Kaiwencheng/DiffSynth-Studio/results/examples/wan/auto_generated/{name}_attn_map_2frame.npy", cache["scaled_dot_product_attention"], allow_pickle=True)
        # get_local.clear()
