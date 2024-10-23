import argparse
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from transformers import TrainerCallback

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


bleu = load_metric("bleu")
rouge = load_metric("rouge")
meteor = load_metric("meteor")


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.compute(
        predictions=decoded_preds, references=[[label] for label in decoded_labels]
    )
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"].mid.fmeasure,
        "rougeL": rouge_score["rougeL"].mid.fmeasure,
        "meteor": meteor_score["meteor"],
    }


class JudgeEvaluationCallback(TrainerCallback):
    def __init__(self, judge_model, judge_tokenizer, eval_dataset):
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs["model_trainer"]
        model = trainer.model
        tokenizer = trainer.tokenizer

        predictions = trainer.predict(self.eval_dataset)
        decoded_preds = tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )

        inputs = self.judge_tokenizer(
            decoded_preds, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.judge_model(**inputs)
        predicted_labels = outputs.logits.argmax(dim=-1)

        num_normal = (predicted_labels == 1).sum().item()
        total = len(predicted_labels)

        print(
            f"Ep {state.epoch}: j-conclusion {num_normal}/{total} ({num_normal / total:.4f})"
        )


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("ridger/MMfreeLM-2.7B")
    model = HGRNBitMultimodalModel(config=HGRNBitMultimodalConfig())

    judge_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    judge_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    dataset = load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1")

    def preprocess_function(ex):
        human_question = next(
            (
                entry["value"]
                for entry in ex["conversations"]
                if entry["from"] == "human"
            ),
            None,
        )
        gpt_answer = next(
            (entry["value"] for entry in ex["conversations"] if entry["from"] == "gpt"),
            None,
        )

        if human_question and gpt_answer and ex.get("video", False):
            inputs = tokenizer_multimodal_token(
                text=human_question,
                video_paths=ex.get("video"),
                audio_paths=None,
                tokenizer=tokenizer,
            )
            inputs["labels"] = tokenizer(gpt_answer, return_tensors="pt")["input_ids"]
            return inputs
        else:
            return {}

    tokenized_datasets = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset["train"].column_names
    )

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
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    judge_callback = JudgeEvaluationCallback(
        judge_model, judge_tokenizer, tokenized_datasets["validation"]
    )
    trainer.add_callback(judge_callback)

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
