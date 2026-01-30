import torch.nn.functional as F

def forward(self, input_ids, attention_mask, labels=None):
      # roberta layer
      output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
      pooled_output = torch.mean(output.last_hidden_state, 1)
      # final logits
      pooled_output = self.dropout(pooled_output)
      pooled_output = self.hidden(pooled_output)
      pooled_output = F.relu(pooled_output)
      pooled_output = self.dropout(pooled_output)
      logits = self.classifier(pooled_output)
      # calculate loss
      loss = 0
      if labels is not None:
        loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
      return loss, logits

import onnx
import torch
import numpy as np
from transformers import AutoTokenizer



onnx_model = onnx.load('ekman-v1-torch (1).onnx')

text = "Text from the news article"

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokens = tokenizer.encode_plus(text,
                    add_special_tokens = True,
                    return_tensors = 'np',
                    truncation = True,
                    max_length = 128,
                    padding = 'max_length',
                    return_attention_mask = True
                    )

input_ids = tokens.input_ids
attention_mask = tokens.attention_mask
labels = np.zeros(6).astype(np.float32)

print(labels)

input ={'input_ids' : input_ids,
        'attention_mask' : attention_mask,
        'labels' : labels} 



onnx.checker.check_model(onnx_model)

import onnxruntime as ort
import numpy as np


ort_sess = ort.InferenceSession('ekman-v1-torch (1).onnx')
outputs = ort_sess.run(None, {'input_ids' : input_ids,'attention_mask' : attention_mask,'labels': labels})

X = next(iter(dummy_dl))
del X["labels"]

print(X)

torch.onnx.export(loaded_model,X,"ekman-v1-torch.onnx",export_params = True, do_constant_folding = True, input_names = ['input_ids','attention_mask','labels'],output_names = [])