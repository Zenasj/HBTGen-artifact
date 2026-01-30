import torch.nn as nn

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    lr_scheduler_type: Optional[str] = field(default="cosine_with_restarts")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    output_dir: Optional[str] = field(default="output")
    remove_unused_columns: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.2-1B-Instruct"
    )
    pad_token: str = field(
        default="<|finetune_right_pad_id|>", metadata={"help": "Padding token."}
    )
    unk_token: str = field(
        default="<|reserved_special_token_0|>",
        metadata={"help": "Unknown token."},
    )


class SupervisedDataset:
    def __init__(self, data, tokenizer, training_args):
        data_dict = SupervisedDataset._preprocess(
            data, tokenizer, training_args
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.classification_labels = [
            d["messages"][-1]["content"] for d in data
        ]

        # Compute class weights for imbalanced classes
        self.class_weights, self.class_values, self.class_indices = (
            SupervisedDataset.get_class_weights(
                self.classification_labels, tokenizer
            )
        )
        self.classification_labels = [
            self.class_indices[label] for label in self.classification_labels
        ]

    @staticmethod
    def get_class_weights(labels, tokenizer):
        classes = sorted(list(set(labels)))
        class_indices = {label: idx for idx, label in enumerate(classes)}
        label_indices = [class_indices[label] for label in labels]

        class_values = []
        for class_name in classes:
            class_values.append(
                tokenizer.encode(class_name, add_special_tokens=False)[0]
            )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_indices),
            y=label_indices,
        )
        return class_weights, class_values, class_indices

    @staticmethod
    def _preprocess(data, tokenizer, training_args):
        formatted_inputs = [
            tokenizer.apply_chat_template(d["messages"], tokenize=False)
            for d in data
        ]
        formatted_prompts = [
            tokenizer.apply_chat_template(
                d["messages"][:-1], tokenize=False, add_generation_prompt=True
            )
            for d in data
        ]
        tokenized_inputs = tokenizer(
            formatted_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_prompts = tokenizer(
            formatted_prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
        )

        attention_mask = tokenized_prompts["attention_mask"]
        input_ids = tokenized_prompts["input_ids"]
        labels = tokenized_inputs["input_ids"][
            :, tokenized_prompts["input_ids"].shape[1]
        ]

        attention_mask = attention_mask[:, -training_args.model_max_length :]
        input_ids = input_ids[:, -training_args.model_max_length :]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def convert_to_dataset(self):
        return Dataset.from_dict(
            {
                "input_ids": self.input_ids,
                "labels": self.labels,
                "attention_mask": self.attention_mask,
            }
        )


# Custom Trainer with weighted loss
class WeightedLoss:
    def __init__(self, class_weights=None, class_values=None):
        self.class_weights = torch.tensor(class_weights).cuda()
        self.class_values = class_values

    def compute_loss(self, outputs, labels, **kwargs):
        logits = outputs.get("logits")

        # Compute loss based on last token logits
        logits = logits[:, -1, self.class_values].reshape(
            -1, len(self.class_values)
        )

        ce_labels = torch.tensor(
            [self.class_values.index(v) for v in labels]
        ).to(labels.device)

        if self.class_weights.dtype != logits.dtype:
            self.class_weights = self.class_weights.to(logits.dtype)

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, ce_labels)

        return loss


# Load and prepare the dataset
def load_and_prepare_data(training_args, tokenizer):
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Please respond with " + ("no" if i % 2 else "yes")
                    ),
                },
                {"role": "assistant", "content": "no" if i % 2 else "yes"},
            ]
        }
        for i in range(1000)
    ]

    dataset = SupervisedDataset(
        dataset,
        tokenizer,
        training_args,
    )

    class_weights, class_values, class_indices = (
        dataset.class_weights,
        dataset.class_values,
        dataset.class_indices,
    )

    dataset = dataset.convert_to_dataset()

    return (
        dataset,
        None,
        class_weights,
        class_values,
        class_indices,
    )


# Training function
def train(model_args, training_args):

    if training_args.lr_scheduler_type == "cosine_with_restarts":
        training_args.lr_scheduler_kwargs = {
            "num_cycles": 1 + training_args.num_train_epochs // 10
        }

    # Load the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # Augment tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_args.pad_token
    if tokenizer.unk_token is None:
        tokenizer.unk_token = model_args.unk_token

    # Load and prepare data
    train_dataset, _, class_weights, class_values, _ = load_and_prepare_data(
        training_args, tokenizer
    )

    # Loss function
    custom_loss = WeightedLoss(class_weights, class_values)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_loss_func=lambda x, y, **kwargs: custom_loss.compute_loss(x, y),
    )

    # Start training
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, CustomTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    train(
        model_args,
        training_args,
    )

grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )