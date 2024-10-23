import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-2.7B")
    model = HGRNBitMultimodalModel(config=HGRNBitMultimodalConfig())

    dataset = load_dataset("lmms-lab/LLaVA-Video-178K")

    def preprocess_function(ex):
        inputs = tokenizer_multimodal_token(
            text=ex["text"],
            video_paths=ex["video_path"],
            audio_paths=ex["audio_path"],  # мало ли
            tokenizer=tokenizer,
        )
        return inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=args.logging_dir,
        logging_steps=10,
        save_steps=1000,
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
