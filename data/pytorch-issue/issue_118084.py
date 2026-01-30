py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

print("Torch version: ", torch.__version__)


def getDefaultDevice():
  if torch.cuda.is_available():
    print("\nSetting default device to cuda.\n")
    torch.set_default_device('cuda')
    print("Device name: ", torch.cuda.get_device_name(device=None))
    print("BF16 support: ", torch.cuda.is_bf16_supported())

    return torch.device('cuda')

  else:
    print("No GPU detected, default device is cpu")

    return torch.device('cpu')

device = getDefaultDevice()


DATASET_ID = 'FinanceInc/auditor_sentiment'
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'

config = {
    'model_type': 'HuggingFace: TinyLlama-1.1B',
    'model_name': MODEL_ID,
    'dataset_type': 'HuggingFace: Multi-class classification',
    'dataset_name': DATASET_ID,
    'n_labels': 3,
    'batch_size': 2,
    'lr': 5e-5,
    'epochs': 5,
    'accumulation_steps': 16,
    'max_length': 128
}

data = load_dataset(config['dataset_name'])
df_train = data['train'].to_pandas()
df_test = data['test'].to_pandas()

class FinanceIncDataset(object):
  def __init__(self, df, tokenizer, max_length=None):
    self.df = df
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token='[PAD]'
    self.tokenizer = tokenizer
    self.n_samples = len(df)
    self.max_len = max_length

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    sentence = self.df.loc[idx, 'sentence']
    label = self.df.loc[idx, 'label']

    tokenizer_output = self.tokenizer.encode_plus(sentence,
                                                  add_special_tokens=True,
                                                  padding='max_length',
                                                  truncation=True,
                                                  max_length=self.max_len,
                                                  return_tensors='pt',
                                                  return_token_type_ids=False,
                                                  return_attention_mask=True)

    input_ids = tokenizer_output['input_ids'].flatten()
    attention_mask = tokenizer_output['attention_mask'].flatten()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label
    }


tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

train_dataset = FinanceIncDataset(df_train, tokenizer, max_length=config['max_length'])
val_dataset = FinanceIncDataset(df_test, tokenizer, max_length=config['max_length'])
df_train = None
df_test = None

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False, generator=torch.Generator(device=device))
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False)

batch = next(iter(train_loader))
print(batch['input_ids'].shape)
print(batch['label'].shape)