from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import torch.nn as nn

model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")

training_arguments = TrainingArguments(output_dir="foo", skip_memory_metrics=True, per_device_train_batch_size=4)

dummy = {}

sequence_length = 128
n_samples = 256

dummy["input_ids"] = torch.randint(
        low=0,
        high=10,
        size=(n_samples, sequence_length))
dummy["attention_mask"] = torch.randint(
        low=0,
        high=2,
        size=(n_samples, sequence_length))

dummy["labels"] = torch.randint(
        low=0,
        high=2,
        size=(n_samples,))

task_dataset = Dataset.from_dict(dummy)
task_dataset.set_format(
    type="torch",  # for now we're using pytorch tensors
    columns=list(task_dataset.features.keys()),
)

train_dataset = task_dataset

print("train_dataset", train_dataset)
print("train_dataset", train_dataset[0])

# Same issue with Transformers Trainer.
"""
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
)
trainer.train()
"""

model = nn.DataParallel(model)
model.to("cuda")

inp = {
    "input_ids": train_dataset[:8]["input_ids"].to("cuda"),
    "attention_mask": train_dataset[:8]["attention_mask"].to("cuda"),
    "labels": train_dataset[:8]["labels"].to("cuda")
}
print("inp", inp)
output = model(**inp)