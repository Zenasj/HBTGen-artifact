import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn

device = torch.device("cuda")

def make_token_tensor(id, vocab_len, should_squeeze=True):
  # print("Making token tensor of id:", id)
  t = torch.zeros(vocab_len).to(device)
  t[id] = 1
  if should_squeeze:
    return t.unsqueeze(0).unsqueeze(0)
  else:
    return t

h_size = 1536 # The Hidden size that goes into the decoder
o_size = 30522 # 30522 = Vocabulary size of default BERT tokenizer

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


decoder = DecoderRNN(h_size, o_size)
decoder.to(device)

# The initial inputs to the decoder
d_hidden = torch.rand((1,1,h_size)).to(device)
prev_token_pred = make_token_tensor(1, o_size) # Has dimensions 1 x 1 x o_size

ans_tokens = [1, 2, 3, 4] # Imagine that in a real model these would be used for teacher forcing
max_len = len(ans_tokens)
seq_preds = []
for i in range(max_len):
  token_pred, d_hidden = decoder(prev_token_pred, d_hidden)
  prev_token_id = torch.argmax(token_pred)
  prev_token_pred = make_token_tensor(prev_token_id, o_size)
  seq_preds.append(token_pred.squeeze(0))
test_preds = torch.stack(seq_preds)

loss = nn.NLLLoss()
input = test_preds
# each element in target has to have 0 <= value < C
target = torch.tensor(ans_tokens).to(device)
output = loss(input, target)
output.backward()