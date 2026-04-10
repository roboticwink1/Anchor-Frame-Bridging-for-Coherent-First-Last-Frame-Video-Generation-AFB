import os
import torch
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video
from modelscope import dataset_snapshot_download
from PIL import Image


# download_models(["HunyuanVideoI2V"])
model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    [
        "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "models/HunyuanVideoI2V/text_encoder/model.safetensors",
        "models/HunyuanVideoI2V/text_encoder_2",
        'models/HunyuanVideoI2V/vae/pytorch_model.pt'
    ],
    torch_dtype=torch.float16,
    device="cuda"
)
# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(model_manager,
                                               torch_dtype=torch.bfloat16,
                                               device="cuda",
                                               enable_vram_management=False)
# Although you have enough VRAM, we still recommend you to enable offload.
pipe.enable_cpu_offload()

# dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#                           local_dir="./",
#                           allow_file_pattern=f"data/examples/hunyuanvideo/*")

selected_data = {
    "DAVIS_25": ["train_000009", "test_000019", "train_000067", "train_000069"],
    "RealEs_25": [x.rjust(6, "0") for x in ["34", "59", "62", "95", "96", "132"]],
    "human_hq_25": [x.rjust(6, "0") for x in ["29", "106", "145", "202", "227", "230", "244", "265"]]
}

names = []
images = []
prompts = []
save_dirs = []

base_prompt_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/data/examples/selected_dataset/first_and_last_frame_base64_results"
base_image_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/data/examples/selected_dataset/the_first_and_last_frame"
base_save_dir = "/public/home/lingwang/lmj/kevin/DiffSynth-Studio/results/examples/hunyuanvideo/auto_generated_new_prompt"

def extract_video_description(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
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

    return ' '.join(description_lines)

for dataset_name, indices in selected_data.items():
    dataset_prompt_dir = os.path.join(base_prompt_dir, dataset_name)
    dataset_image_dir = os.path.join(base_image_dir, dataset_name)
    dataset_save_dir = os.path.join(base_save_dir, dataset_name, "reverse")
    os.makedirs(dataset_save_dir, exist_ok=True)

    for i in indices:
        image_dir = os.path.join(dataset_image_dir, i)
        prompt_dir = os.path.join(dataset_prompt_dir, i)
        names.append(f"{dataset_name}-{i}")

        images.append(
            [
                Image.open(os.path.join(image_dir, "image24.png")).convert("RGB"),
                Image.open(os.path.join(image_dir, "image00.png")).convert("RGB")
            ]
        )

        prompts.append(
            extract_video_description(os.path.join(prompt_dir, "video_metadata_new_reverse.md"))
        )

        save_dirs.append(
            os.path.join(dataset_save_dir, f"{i}.mp4")
        )


i2v_resolution = "720p"
# middle_frame_position = 16  # Set this between [1, 31]

for name, imgs, prompt, dir in zip(names, images, prompts, save_dirs):
    print(f"Reverse generating {name}...")
    print(f"Detected {len(imgs)} input images.")
    assert len(imgs) in {1, 2, 3}, "Only support 1 or 2 or 3 input key frame(s)."
    video = pipe(prompt, input_images=imgs, num_inference_steps=50, seed=0, i2v_resolution=i2v_resolution, num_frames=81)
    save_video(video, dir, fps=30, quality=6)
