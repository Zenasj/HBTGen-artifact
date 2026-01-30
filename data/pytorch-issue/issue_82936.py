import torch.nn as nn

from flask import Flask
import sys
import sys
import json
import nltk
import os
import random
import re
from collections import Counter
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle

app = Flask(__name__)

with open("cutoff.txt", "rb") as f:
    filtered_words = pickle.load(f)
with open('vocab.txt', 'rb') as f:
    vocab = pickle.load(f)
def predict(text, model, vocab):
    tokens = preprocess(text)
    tokens = [word for word in tokens if word in filtered_words]
    tokens = [vocab[word] for word in tokens]
    text_input = torch.tensor(tokens).unsqueeze(1).cuda()
    hidden = model.init_hidden(text_input.size(1))
    logps, _ = model.forward(text_input.to(torch.int64),hidden)
    pred = torch.exp(logps.cpu()).detach().numpy()
    return pred
nltk.download('wordnet')
nltk.download('omw-1.4')
def preprocess(message):
    text = message.lower()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text)
    text = re.sub (r'^$.*\s',' ',text)
    text = re.sub (r'^@.*\s',' ',text)
    text = re.sub (r'[^a-zA-Z]',' ',text)
    tokens = text.split()
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens if len(t) > 1]
    return tokens

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.embed_size)
        self.lstm = nn.LSTM(input_size = self.embed_size,hidden_size = self.lstm_size,num_layers  = self.lstm_layers,batch_first = False,dropout = self.dropout)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_features = self.lstm_size, out_features= self.output_size)
        self.log_smax = nn.LogSoftmax(dim=1)
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden_state = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(),
                      weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        return hidden_state
    def forward(self, nn_input, hidden_state):
        nn_input.cuda()
        embed = self.embedding(nn_input)
        lstm_out, hidden_state = self.lstm(embed, hidden_state)
        lstm_out = lstm_out[-1]
        logps = self.log_smax(self.dropout(self.fc(lstm_out)))
        return logps, hidden_state
wmodel = torch.load("shuja_sentiment.dvi",map_location=torch.device('cuda'))

model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
model.load_state_dict(wmodel.state_dict())

@app.route("/ticker/<string:tweet>", methods=["GET"])
def ticker(tweet):
    model.eval()
    model.to("cuda")
    Forecast = predict(tweet, model, vocab)
    return "class {} is the most probable class with a likelihood of {:.2f}%".format(np.argmax(Forecast) - 2,
                                                                                    np.max(Forecast) * 100)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001)