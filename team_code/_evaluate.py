import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from functools import partial
from typing import Optional, Dict, Any

import torch
from transformers.utils import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from mm_utils import (
    process_image,
    process_video,
    process_audio,
    tokenizer_multimodal_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from constants import (
    NUM_FRAMES,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_AUDIO_TOKEN,
    MODAL_INDEX_MAP,
)

from omnimmfreecore.config import HGRNBitMultimodalConfig
from omnimmfreecore.modeling_hgrn_multimodal_bit import HGRNBitMultimodalModel

logging.set_verbosity("ERROR")

print(f"TRANSFORMERS_OFFLINE={os.environ['TRANSFORMERS_OFFLINE']}")


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def setup_model_and_tokenizer(
    device_map: Optional[str] = "auto",
    device: Optional[str] = "cuda",
    use_flash_attn: Optional[bool] = False,
    **kwargs,
):

    model_path = "/home/jovyan/models/hgrnbitmultimodal"  # TODO PATH2MODEL

    if not os.path.exists(model_path):
        raise FileExistsError(f"Model's checkpoint not found at path: {model_path}")

    kwargs = {"device_map": device_map, **kwargs}
    if not device.startswith("cuda"):
        kwargs["device_map"] = {"": device}
    kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    config = HGRNBitMultimodalConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = HGRNBitMultimodalModel.from_pretrained(
        model_path, low_cpu_mem_usage=True, config=config, **kwargs
    )

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = (
        model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    )
    print(f"Running with num_frames={num_frames}")

    processor = {
        "image": partial(process_image, aspect_ratio=None),
        "video": partial(process_video, aspect_ratio=None, num_frames=num_frames),
        "audio": partial(process_audio, processor=None),  # без аудио
    }

    return model, processor, tokenizer


def process_data_sample(
    sample: Dict[str, Any], processor, modality_type: Optional[str] = "video", **kwargs
) -> Dict[str, Any]:
    modality_path = sample.get(modality_type)
    if not modality_path or not os.path.exists(modality_path):
        raise FileNotFoundError(f"Can't find file by path: `{modality_path}`")

    modality_features = processor[modality_type](modality_path)

    if sample["task_type"] == "qa":
        question = sample["question"]
        choices = [choice["choice"] for choice in sample["choices"]]
        options = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        instruction = f"Question: {question}\nOptions:" + "\n".join(
            f"{opt} {ans}" for opt, ans in zip(options, choices)
        )

        instruction += "\nAnswer with the option's letter."

        return {
            "task_id": sample["task_id"],
            "task_type": sample["task_type"],
            modality_type: modality_features,
            "instruction": instruction,
            "answers": [f"{o} {a}" for o, a in zip(options, choices)],
        }

    elif sample["task_type"] == "captioning":
        instruction = f"Question: {sample['question']}\nAnswer: "
        return {
            "task_id": sample["task_id"],
            "task_type": sample["task_type"],
            modality_type: modality_features,
            "instruction": instruction,
        }


def evaluate_generative(
    sample: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer,
    modality_type: Optional[str] = "video",
    **kwargs,
) -> str:
    modal_token = {
        "image": DEFAULT_IMAGE_TOKEN,
        "video": DEFAULT_VIDEO_TOKEN,
        "audio": DEFAULT_AUDIO_TOKEN,
        "text": "",
    }.get(modality_type, "")

    modality_features = sample.get(modality_type)
    tensor = (
        modality_features.half().to(model.device) if modality_type != "text" else None
    )
    tensor = [(tensor, modality_type)] if tensor else None

    question = modal_token + "\n" + sample["instruction"]

    input_ids = (
        tokenizer_multimodal_token(
            question, tokenizer, modal_token, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(model.device)
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(model.device)

    keywords = [tokenizer.eos_token]
    temperature = kwargs.get("temperature", 0.2)
    top_p = kwargs.get("top_p", 0.9)
    max_new_tokens = kwargs.get("max_new_tokens", 2048)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=kwargs.get("do_sample", False),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[
                KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            ],
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# МУСОР ЕБАНЫЙ БЛЯТЬ
""" def evaluate_ppl(
    sample: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer,
    modality_type: Optional[str] = "video",
    **kwargs,
) -> int: """
