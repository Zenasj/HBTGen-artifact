from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments, HfArgumentParser
import pandas as pd
import numpy as np
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_DEBUG']='INFO'
os.environ['NCCL_DEBUG_SUBSYS']='ALL'
os.environ['NCCL_IB_DISABLE']='1'
os.environ['NCCL_SOCKET_IFNAME']='eth0'

tok = RobertaTokenizerFast.from_pretrained('/home/jovyan/models/roberta-large/')
model = RobertaForSequenceClassification.from_pretrained('/home/jovyan/models/roberta-large/', num_labels=2)

df_full = pd.read_csv('IMDB_Dataset.csv')
print("loaded df")
df_full = df_full.sample(frac=1).reset_index(drop=True)
df_req =  df_full.head(1000)
df_train = df_req.head(800)
df_eval = df_req.tail(200)
train_text, train_labels_raw, val_text, val_labels_raw = df_train.review.values.tolist(), df_train.sentiment.values.tolist(), df_eval.review.values.tolist(), df_eval.sentiment.values.tolist(),


train_encodings = tok(train_text, padding=True, truncation=True, max_length=512)
val_encodings = tok(val_text, padding=True, truncation=True, max_length=512)
train_labels = [1 if i=='positive' else 0 for i in train_labels_raw]
val_labels = [1 if i=='positive' else 0 for i in val_labels_raw]


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

print("Encoding done")


parser = HfArgumentParser(TrainingArguments)
train_args = parser.parse_args_into_dataclasses()
print('parser and args created')


trainer = Trainer(
             model=model,
             args=train_args[0],
             train_dataset=train_dataset,
             eval_dataset=val_dataset
             )
if train_args[0].do_train:
    print('------------TRAINING-------------')
    trainer.train() 
if train_args[0].do_eval:
    print('------------EVALUATING-------------')
    trainer.evaluate()