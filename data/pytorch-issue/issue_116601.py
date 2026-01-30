if torch.backends.mps.is_available():
    device = torch.device('mps')

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random

# Tokenizer for English and German
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def yield_tokens(data_iter: Iterable, language, ind) -> Iterable:
    for _, data in enumerate(data_iter):
        yield language(data[ind])

# Load dataset
train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en','de'))

# Build vocab
vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_en, 0), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_de, 1), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_en.set_default_index(vocab_en["<unk>"])
vocab_de.set_default_index(vocab_de["<unk>"])

# Text pipeline functions
text_pipeline_en = lambda x: [vocab_en["<sos>"]] + [vocab_en[token] for token in tokenizer_en(x)] + [vocab_en["<eos>"]]
text_pipeline_de = lambda x: [vocab_de["<sos>"]] + [vocab_de[token] for token in tokenizer_de(x)] + [vocab_de["<eos>"]]

# Padding function
def collate_batch(batch):
    src_list, trg_list = [], []
    for src_sample, trg_sample in batch:
        src_list.append(torch.tensor(text_pipeline_en(src_sample.rstrip("\n")), dtype=torch.int64))
        trg_list.append(torch.tensor(text_pipeline_de(trg_sample.rstrip("\n")), dtype=torch.int64))
    src_list = pad_sequence(src_list, padding_value=vocab_en["<pad>"])
    trg_list = pad_sequence(trg_list, padding_value=vocab_de["<pad>"])
    return src_list, trg_list

# Define the data loaders
BATCH_SIZE = 128
device = torch.device('mps')

train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Hyperparameters and model instantiation
INPUT_DIM = len(vocab_en)
OUTPUT_DIM = len(vocab_de)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

# Training
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = vocab_de["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / i

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / i

N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Evaluation
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

# Tokenizer for English and German
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def yield_tokens(data_iter: Iterable, language, ind) -> Iterable:
    for _, data in enumerate(data_iter):
        yield language(data[ind])

# Load dataset
train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en','de'))

# Build vocab
vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_en, 0), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_de, 1), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_en.set_default_index(vocab_en["<unk>"])
vocab_de.set_default_index(vocab_de["<unk>"])

# Text pipeline functions
text_pipeline_en = lambda x: [vocab_en["<sos>"]] + [vocab_en[token] for token in tokenizer_en(x)] + [vocab_en["<eos>"]]
text_pipeline_de = lambda x: [vocab_de["<sos>"]] + [vocab_de[token] for token in tokenizer_de(x)] + [vocab_de["<eos>"]]

# Padding function
def collate_batch(batch):
    src_list, trg_list = [], []
    for src_sample, trg_sample in batch:
        src_list.append(torch.tensor(text_pipeline_en(src_sample.rstrip("\n")), dtype=torch.int64))
        trg_list.append(torch.tensor(text_pipeline_de(trg_sample.rstrip("\n")), dtype=torch.int64))
    src_list = pad_sequence(src_list, padding_value=vocab_en["<pad>"])
    trg_list = pad_sequence(trg_list, padding_value=vocab_de["<pad>"])
    return src_list, trg_list

# Define the data loaders
BATCH_SIZE = 128
device = torch.device('mps')

train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):        
        embedded = self.dropout(self.embedding(src))        
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))        
        return outputs, hidden
    

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, hidden.squeeze(0)
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = src.permute(1, 0)
        # trg = trg.permute(1, 0)
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
                
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top_one = output.argmax(1) 
            input = trg[t] if random.random() < teacher_forcing_ratio else top_one

        return outputs

# Hyperparameters and model instantiation
INPUT_DIM = len(vocab_en)
OUTPUT_DIM = len(vocab_de)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT)
att = Attention(HID_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, att)

model = Seq2Seq(enc, dec, device).to(device)

# Training
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = vocab_de["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / i

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / i

N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Evaluation
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

# Tokenizer for English and German
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')

def yield_tokens(data_iter: Iterable, language, ind) -> Iterable:
    for _, data in enumerate(data_iter):
        yield language(data[ind])

# Load dataset
train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair=('en','de'))

# Build vocab
vocab_en = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_en, 0), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_de = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer_de, 1), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_en.set_default_index(vocab_en["<unk>"])
vocab_de.set_default_index(vocab_de["<unk>"])

# Text pipeline functions
text_pipeline_en = lambda x: [vocab_en["<sos>"]] + [vocab_en[token] for token in tokenizer_en(x)] + [vocab_en["<eos>"]]
text_pipeline_de = lambda x: [vocab_de["<sos>"]] + [vocab_de[token] for token in tokenizer_de(x)] + [vocab_de["<eos>"]]

# Padding function
def collate_batch(batch):
    src_list, trg_list = [], []
    for src_sample, trg_sample in batch:
        src_list.append(torch.tensor(text_pipeline_en(src_sample.rstrip("\n")), dtype=torch.int64))
        trg_list.append(torch.tensor(text_pipeline_de(trg_sample.rstrip("\n")), dtype=torch.int64))
    src_list = pad_sequence(src_list, padding_value=vocab_en["<pad>"])
    trg_list = pad_sequence(trg_list, padding_value=vocab_de["<pad>"])
    return src_list, trg_list

# Define the data loaders
BATCH_SIZE = 128
device = torch.device('mps')

train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)

# Seq2Seq Model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, 1, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):        
        embedded = self.dropout(self.embedding(src))        
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))        
        return outputs, (hidden, cell)
    

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim, 1)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        return prediction, (hidden.squeeze(0), cell.squeeze(0))
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = src.permute(1, 0)
        # trg = trg.permute(1, 0)
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
                
        input = trg[0, :]
        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top_one = output.argmax(1) 
            input = trg[t] if random.random() < teacher_forcing_ratio else top_one

        return outputs

# Hyperparameters and model instantiation
INPUT_DIM = len(vocab_en)
OUTPUT_DIM = len(vocab_de)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, HID_DIM, ENC_DROPOUT)
att = Attention(HID_DIM, HID_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, HID_DIM, DEC_DROPOUT, att)

model = Seq2Seq(enc, dec, device).to(device)

# Training
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = vocab_de["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / i

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / i

N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

# Evaluation
test_loss = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')