import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput

class ReproError(nn.Module):

    def forward(self, inputs):
        return Seq2SeqLMOutput(logits=inputs)


def main():
    model = torch.compile(ReproError())
    model(torch.tensor([0.1, 0.2]))

main()

hf_model.generate2 = torchdynamo.optimize("inductor")(hf_model.generate)

torch.set_float32_matmul_precision('high')