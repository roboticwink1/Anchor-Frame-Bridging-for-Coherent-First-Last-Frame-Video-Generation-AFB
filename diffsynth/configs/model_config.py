from typing_extensions import Literal, TypeAlias

# HunyuanVideo models
from ..models.hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from ..models.hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..models.sd3_text_encoder import SD3TextEncoder1

# Wan Video models
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel


model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    
    # SD3 Text Encoder (used by HunyuanVideo)
    (None, "94eefa3dac9cec93cb1ebaf1747d7b78", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    (None, "5da81baee73198a7c19e6d2fe8b5148e", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    
    # HunyuanVideo
    (None, "aeb82dce778a03dcb4d726cb03f3c43f", ["hunyuan_video_vae_decoder", "hunyuan_video_vae_encoder"], [HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder], "diffusers"),
    (None, "b9588f02e78f5ccafc9d7c0294e46308", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    (None, "84ef4bd4757f60e906b54aa6a7815dc6", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    
    # Wan Video
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
]

huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("LlamaForCausalLM", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoLLMEncoder"),
    ("LlavaForConditionalGeneration", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoMLLMEncoder"),
]

patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
]

preset_models_on_huggingface = {
    # HunyuanVideo
    "HunyuanVideo": {
        "file_list": [
            ("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors", "models/HunyuanVideo/text_encoder"),
        ],
        "load_path": [],
    },
}

preset_models_on_modelscope = {
    # HunyuanVideo
    "HunyuanVideo": {
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideoI2V": {
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideoI2V/text_encoder"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00001-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00002-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00003-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00004-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "config.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model.safetensors.index.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/vae/pytorch_model.pt", "models/HunyuanVideoI2V/vae"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideoI2V/transformers")
        ],
        "load_path": [
            "models/HunyuanVideoI2V/text_encoder/model.safetensors",
            "models/HunyuanVideoI2V/text_encoder_2",
            "models/HunyuanVideoI2V/vae/pytorch_model.pt",
            "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideo-fp8": {
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("DiffSynth-Studio/HunyuanVideo-safetensors", "model.fp8.safetensors", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/model.fp8.safetensors"
        ],
    },
}

Preset_model_id: TypeAlias = Literal[
    "HunyuanVideo",
    "HunyuanVideo-fp8",
    "HunyuanVideoI2V",
]
