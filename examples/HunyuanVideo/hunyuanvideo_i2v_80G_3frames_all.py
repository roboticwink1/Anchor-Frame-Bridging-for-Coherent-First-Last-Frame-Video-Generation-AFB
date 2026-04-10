import os
import torch
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video
from modelscope import dataset_snapshot_download
from PIL import Image


# download_models(["HunyuanVideoI2V"])
model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    ["models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"],
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "models/HunyuanVideoI2V/text_encoder/model.safetensors",
        "models/HunyuanVideoI2V/text_encoder_2",
        "models/HunyuanVideoI2V/vae/pytorch_model.pt",
    ],
    torch_dtype=torch.float16,
    device="cuda",
)
# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda",
    enable_vram_management=False,
)
# Although you have enough VRAM, we still recommend you to enable offload.
pipe.enable_cpu_offload()

# dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#                           local_dir="./",
#                           allow_file_pattern=f"data/examples/hunyuanvideo/*")

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
base_mid_frame_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/data/examples/selected_dataset/middle_frame_hunyuan_all"
base_save_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/results/examples/hunyuanvideo_all/mid_auto_generated"


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
    dataset_mid_frame_dir = os.path.join(base_mid_frame_dir, dataset_name)
    dataset_save_dir = os.path.join(base_save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    indices = list_one_level_subdirs(dataset_image_dir)
    indices.sort()

    for i in indices:
        if not os.path.exists(os.path.join(dataset_mid_frame_dir, f"{os.path.basename(i)}.png")):
            print(f"{dataset_name}-{i} is missing.")
            continue
        image_dir = os.path.join(dataset_image_dir, i)
        names.append(f"{dataset_name}-{i}")

        images.append(
            [
                Image.open(os.path.join(image_dir, "image00.png")).convert("RGB"),
                Image.open(os.path.join(dataset_mid_frame_dir, f"{os.path.basename(i)}.png")).convert(
                    "RGB"
                ),
                Image.open(os.path.join(image_dir, "image24.png")).convert("RGB"),
            ]
        )

        prompts.append(
            extract_video_description(os.path.join(image_dir, "video_metadata_old.md"))
        )

        save_dirs.append(os.path.join(dataset_save_dir, f"{os.path.basename(i)}.mp4"))


i2v_resolution = "720p"
# middle_frame_position = 16  # Set this between [1, 31]
print("Total number of videos to generate:", len(names))

for i, (name, imgs, prompt, dir) in enumerate(zip(names, images, prompts, save_dirs)):
    if i > 60:
        print(f"AFC generating {name}...")
        print(f"Detected {len(imgs)} input images.")
        assert len(imgs) in {1, 2, 3}, "Only support 1 or 2 or 3 input key frame(s)."
        num_frames = 81
        video = pipe(
            prompt,
            input_images=imgs,
            middle_frame_position=int(((num_frames - 1) // 4 + 1) * (3 / 4)),
            num_inference_steps=50,
            seed=0,
            i2v_resolution=i2v_resolution,
            num_frames=num_frames,
        )
        save_video(video, dir, fps=30, quality=6)
