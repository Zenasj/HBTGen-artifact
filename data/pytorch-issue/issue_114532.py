import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from io import StringIO
from transformers import TrainingArguments, Trainer

sample_data = """input,labels
bazinga,0
please-just-work,1
"""

df = pd.read_csv(StringIO(sample_data))
ds = Dataset.from_pandas(df)

model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenizer_func(x): return tokenizer(x["input"])
ds_tokenized = ds.map(tokenizer_func, batched=True)
dds = ds_tokenized.train_test_split(0.2, seed=42)

bs = 16
epochs = 4
lr = 8e-5

args = TrainingArguments(
    "outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    evaluation_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
trainer = Trainer(
    model,
    args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],
    tokenizer=tokenizer,
)

# This crashes
trainer.train()