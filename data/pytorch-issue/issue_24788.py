import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seed = 23
np.random.seed(seed)
torch.manual_seed(seed)

hidden_size = 256
output_lang_vocab = 2803
max_length = 10
seq_len = max_length
loop_length = 15

stop_id = 658
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: [1,1]
        # hidden: [1,1,256]
        # encoder_outputs: [seq_len, 256], say seq_len = 10

        embedded = self.embedding(input).view(1, 1, -1)  # [1, 1, 256]
        embedded = self.dropout(embedded)
        tmp = torch.cat((embedded[0], hidden[0]), 1)  # [1, 512]

        tmp2 = self.attn(tmp)  # [1, max_length]
        attn_weights = F.softmax(tmp2, dim=1)  # [1, max_length]

        attn_weights_r = attn_weights.unsqueeze(0)  # [1, 1, max_length]
        encoder_outputs_r = encoder_outputs.unsqueeze(0)  # [1, seq_len, 256]

        attn_applied = torch.bmm(attn_weights_r, encoder_outputs_r)  # [1, 1, 256]

        embed0 = embedded[0]  # [1, 256]
        attn0 = attn_applied[0]  # [1, 256]

        output = torch.cat((embed0, attn0), 1)  # [1, 512]
        output_tmp = self.attn_combine(output)  # [1, 256]
        output = output_tmp.unsqueeze(0)  # [1, 1, 256]

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)  # [1, 1, 256]

        # output = F.log_softmax(self.out(output[0]), dim=1) #[1, 2803]
        output = self.out(output[0]) #[1, 2803]

        return output, hidden, attn_weights


class GreedyDecoder(nn.Module):
    def __init__(self, hidden_size, output_lang_vocab, loop_length=15, stop_id=1):
        super(GreedyDecoder, self).__init__()
        self.decoder = AttnDecoderRNN(hidden_size, output_lang_vocab, dropout_p=0.1)
        self.loop_length = loop_length
        self.stop_id = stop_id

    def forward(self, decoder_input, decoder_hidden, encoder_outputs_t):
        decoded_words_ids = torch.zeros([self.loop_length], dtype=torch.int32)

        for di in range(self.loop_length):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs_t)

            topv, topi = torch.topk(decoder_output, 1)
            index = torch.tensor([di])
            data = topi.int().squeeze(0)
            decoded_words_ids.scatter_(0, index, data)
            if int(topi) == 1 or int(topi) == self.stop_id:
                break

            decoder_input = topi.squeeze().detach()

        return decoded_words_ids


decoder_input = torch.tensor([[0]], device=device)  # SOS
decoder_hidden = torch.from_numpy(np.random.rand(1, 1, hidden_size).astype(np.float32))
encoder_outputs_t = torch.from_numpy(np.random.rand(max_length, hidden_size).astype(np.float32))

model = GreedyDecoder(hidden_size, output_lang_vocab, loop_length, stop_id)
model.eval()

output_ids = model(decoder_input, decoder_hidden, encoder_outputs_t)
print(len(output_ids))
print(output_ids)

scripted_model = torch.jit.script(model)

out_from_script = scripted_model(decoder_input, decoder_hidden, encoder_outputs_t)
print(out_from_script)

torch.onnx.export(scripted_model, (decoder_input, decoder_hidden, encoder_outputs_t),
                  'greedy_decoder.onnx', verbose=True,
                  example_outputs = (output_ids))