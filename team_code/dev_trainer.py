import argparse
import datasets
import torch

from transformers import TrainerCallback
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
import evaluate  # type: ignore
from sklearn.model_selection import train_test_split
from mm_utils import tokenizer_multimodal_token
from omnimmfreecore.modeling_hgrn_multimodal_bit import HGRNBitMultimodalModel
from omnimmfreecore.config import HGRNBitMultimodalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="train config")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size/device")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_dir", type=str, default="./logs")
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


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-2.7B")
    model = HGRNBitMultimodalModel(config=HGRNBitMultimodalConfig())

    dataset = datasets.load_dataset(
        "lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1", split="multi_choice"
    )

    def preprocess_function(ex):
        conversations = ex.get("conversations", [])
        if not isinstance(conversations, list):
            return {}

        question = next(
            (
                entry["value"]
                for entry in conversations
                if isinstance(entry, dict) and entry.get("from") == "human"
            ),
            None,
        )
        correct_answer = next(
            (
                entry["value"]
                for entry in conversations
                if isinstance(entry, dict) and entry.get("from") == "gpt"
            ),
            None,
        )

        if question and correct_answer and ex.get("video"):
            inputs = tokenizer_multimodal_token(
                text=question,
                video_paths=ex.get("video"),
                audio_paths=None,
                tokenizer=tokenizer,
            )
            inputs["labels"] = tokenizer(correct_answer, return_tensors="pt")[
                "input_ids"
            ]
            return inputs
        else:
            return {}

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_Tdataset, test_Tdataset = train_test_split(tokenized_datasets, test_size=0.3)

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
        dataloader_num_workers=4,
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
