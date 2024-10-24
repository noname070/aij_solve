import argparse
import os
from typing import List, Tuple, Union
import cv2
import imageio
from functools import partial
from PIL import Image
import tempfile
import numpy as np
import torch
from decord import VideoReader, cpu  # type: ignore
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoTokenizer, Trainer, TrainingArguments
import evaluate  # type: ignore
import datasets
from datasets import DownloadConfig

from mm_utils import tokenizer_multimodal_token, expand2square, frame_sample

from omnimmfreecore.modeling_hgrn_multimodal_bit import HGRNBitMultimodalModel
from omnimmfreecore.config import HGRNBitMultimodalConfig

from constants import NUM_FRAMES, MAX_FRAMES


def parse_args():
    parser = argparse.ArgumentParser(description="train config")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, default="/workspace/aij_solve/outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size/device")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_dir", type=str, default="/workspace/aij_solve/runs")
    return parser.parse_args()


accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
precision_metric = evaluate.load("precision", trust_remote_code=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return {"accuracy": accuracy["accuracy"], "precision": precision["precision"]}


def process_image(image_path, processor, aspect_ratio="pad"):
    images = [Image.open(image_path).convert("RGB")]

    if aspect_ratio == "pad":
        images = [
            expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            for image in images
        ]
    else:
        images = [Image.fromarray(f) for f in images]

    images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images


def process_video(
    video_paths,
    video_dataset,
    vprocessor=None,
    aspect_ratio="pad",
    num_frames=NUM_FRAMES,
):
    s = None
    e = None
    videos = []
    print(f"processing {len(videos)} videos")
    for vpath in video_paths:
        vid = video_dataset["__key__"].index(vpath.split(".")[0])

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as temp_video_file:
            temp_video_file.write(video_dataset["mp4"][vid])
            temp_video_path = temp_video_file.name

        print(f"loading in VideoReader {temp_video_path} ")
        vreader = VideoReader(temp_video_path, ctx=cpu(0), num_threads=1)
        print(f"loaded {temp_video_path} ")
        fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

        f_start = 0 if s is None else max(int(s * fps) - 1, 0)
        f_end = (
            num_frames_of_video - 1
            if e is None
            else min(int(e * fps) - 1, num_frames_of_video - 1)
        )
        frame_indices = list(range(f_start, f_end + 1))
        duration = len(frame_indices)

        if num_frames is None:
            sampled_frame_indices = [
                frame_indices[i] for i in frame_sample(duration, mode="fps", fps=fps)
            ]
        else:
            sampled_frame_indices = [
                frame_indices[i]
                for i in frame_sample(duration, mode="uniform", num_frames=num_frames)
            ]

        video_data = [
            Image.fromarray(frame)
            for frame in vreader.get_batch(sampled_frame_indices).asnumpy()
        ]

        while num_frames is not None and len(video_data) < num_frames:
            video_data.append(
                Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8))
            )

        video_data = video_data[:MAX_FRAMES]
        images = [f for f in video_data]
        video = vprocessor.preprocess(images, return_tensors="pt")["pixel_values"]
        videos.append(video)
        os.remove(temp_video_path)
        print(f"done")

    return torch.Tensor(videos)


def process_audio(audio, processor):
    return processor(audio) if processor else None


def setup_model_and_tokenizer() -> (
    Tuple[
        transformers.AutoModel, transformers.AutoProcessor, transformers.AutoTokenizer
    ]
):
    model = HGRNBitMultimodalModel(config=HGRNBitMultimodalConfig())
    tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-2.7B")

    vision_tower = model.vision_tower.to(
        device=torch.device("cuda"), dtype=torch.float16
    )

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = (
        model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    )

    processor = {
        "image": partial(
            process_image, processor=vision_tower.image_processor, aspect_ratio=None
        ),
        "video": partial(
            process_video,
            vprocessor=vision_tower.image_processor,
            aspect_ratio=None,
            num_frames=num_frames
        ),
        "audio": partial(process_audio, processor=None),
    }

    return model, processor, tokenizer


def main():
    args = parse_args()

    model, processors, tokenizer = setup_model_and_tokenizer()


    print("loading dataset")
    dataset = datasets.load_dataset(
        "lmms-lab/LLaVA-Video-178K",
        "0_30_s_academic_v0_1",
        split="multi_choice",
        download_config=DownloadConfig(resume_download=True, extract_on_the_fly=True),
        # data_dir="0_30_s_academic_v0_1"
    )

    print("loading v_dataset")
    video_dataset = datasets.load_dataset(
        "lmms-lab/LLaVA-Video-178K",
        "0_30_s_academic_v0_1",
        split="train",
        download_config=DownloadConfig(resume_download=True, extract_on_the_fly=True),
        data_dir="0_30_s_academic_v0_1",
    )

    def preprocess(data):
        print("starting preprocessing data")
        conversation = data["conversations"]

        print("preprocessing prompts")
        prompt = next(
            (
                msg["value"].strip().replace("<image>\n", "")
                for msg in conversation
                if isinstance(msg, dict) and msg.get("from") == "human"
            ),
            "",
        )
        print("preprocessing responses")
        response = next(
            (
                msg["value"].strip()
                for msg in conversation
                if isinstance(msg, dict) and msg.get("from") == "gpt"
            ),
            "",
        )

        print("tokenizing inputs")
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        print("tokenizing labels")
        label_inputs = tokenizer(
            response,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        print("processing video")
        video_tensor = processors["video"](data["video"], video_dataset)

        return {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "video_tensor": video_tensor,
            "labels": label_inputs["input_ids"].squeeze(0),
            "label_attention_mask": label_inputs["attention_mask"].squeeze(0),
        }

    print(f"Raw dataset size: {len(dataset)}")
    print(f"dataset column_names : {dataset.column_names}")
    print(f"dataset example : {dataset[0]}")
    print(f"video_dataset column_names : {video_dataset.column_names}")
    print(
        f"video_dataset example : {video_dataset['__key__'][0][:150]} | {video_dataset['mp4'][0][:150]}"
    )
    
    print("start ds mapping preprocessor")
    tokenized_datasets = dataset.map(preprocess, batched=True, num_proc=4)

    train_test_data = tokenized_datasets.train_test_split(test_size=0.3)
    train_Tdataset = train_test_data["train"]
    test_Tdataset = train_test_data["test"]

    print(f"Train dataset size: {len(train_Tdataset)}")
    print(f"Test dataset size: {len(test_Tdataset)}")

    print(f"Train example: {train_Tdataset[0]} ")
    print(f"Test example: {train_Tdataset[0]} ")

    # TEST ONLY
    for i in range(len(tokenized_datasets)):
        try:
            _ = tokenized_datasets[i]
        except KeyError as e:
            print(f"KeyError at index {i}: {e}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        do_train=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=args.logging_dir,
        logging_steps=10,
        save_steps=100,
        load_best_model_at_end=True,
        save_total_limit=5,
        report_to="tensorboard",
        fp16=True,
        dataloader_num_workers=0,
        gradient_accumulation_steps=2,
        deepspeed=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_Tdataset,
        eval_dataset=test_Tdataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
