import torch
import torch.nn as nn

class OnnxDecoder(nn.Module):
    def __init__(self, *, feat_in, num_classes, init_mode="xavier_uniform"):
        super().__init__()

        self._feat_in = feat_in
        self._num_classes = num_classes

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        o = self.decoder_layers(encoder_output).transpose(1, 2)
        return functional.log_softmax(o, dim=-1)


class OnnxModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.decoder = OnnxDecoder(
            feat_in=1024,
            num_classes=vocab_size,
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)[:, :1024]
        x = x.view(-1, 1024, 1)
        return self.decoder(x)

class OnnxDecoder(torch.jit.ScriptModule): # modification is here
    def __init__(self, *, feat_in, num_classes, init_mode="xavier_uniform"):
        super().__init__()

        self._feat_in = feat_in
        self._num_classes = num_classes

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        o = self.decoder_layers(encoder_output).transpose(1, 2)
        return functional.log_softmax(o, dim=-1)


class OnnxModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.decoder = OnnxDecoder(
            feat_in=1024,
            num_classes=vocab_size,
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)[:, :1024]
        x = x.view(-1, 1024, 1)
        return self.decoder(x)