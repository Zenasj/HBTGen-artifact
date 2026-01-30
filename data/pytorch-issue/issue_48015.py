import numpy as np

import torch
model = torch.jit.load('traced_bert.pt', map_location=torch.device('cpu'))

from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('tokenizer/',do_lower_case=True)

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 256

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]

print('input_ids done')

## Create attention mask
attention_masks = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
attention_masks = [[float(i>0) for i in seq] for seq in input_ids]

print('attention_masks done')

# convert all our data into torch tensors, required data type for our model
inputs = torch.tensor(input_ids)
masks = torch.tensor(attention_masks)

print('input and mask tensors done')
print('model ready')

input_id = inputs
input_mask = masks

print('inputs ready')

with torch.no_grad():
    # Forward pass, calculate logit predictions
    with torch.jit.optimized_execution(True, {'target_device': 'eia:0'}):
        print('creating logits')
        logits = model(input_id, attention_mask=input_mask)[0]
        print('logits done')

logits = logits.to('cpu').numpy()

pred_flat = np.argmax(logits, axis=1).flatten()