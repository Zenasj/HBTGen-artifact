import torch.nn as nn

torch/nn/modles/rnn.py

self.rnn = nn.Sequential(
            BidirectionalLSTM(512, self.model_config.n_hiddens, self.model_config.n_hiddens),
            BidirectionalLSTM(self.model_config.n_hiddens, self.model_config.n_hiddens, self.model_config.n_classes))

self.runn(conv)