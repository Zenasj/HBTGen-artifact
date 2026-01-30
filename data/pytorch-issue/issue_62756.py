import torch.nn as nn

import torch
from transformers import *

from transformers import LEDTokenizer, LEDForConditionalGeneration
class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", torchscript=True).led.encoder.cuda()
    def forward(self, data):
        return self.model(data.cuda())

example = torch.zeros((1,128), dtype=torch.long) # bsz , seqlen
pt_model = WrappedModel().eval()
traced_script_module = torch.jit.trace(pt_model, example)
# traced_script_module.save("traced_model.pt")
example_concurrent_batch = torch.zeros((4,128), dtype=torch.long) # bsz , seqlen
traced_script_module(example_concurrent_batch)

class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', torchscript=True).cuda()
    def forward(self, data):
        return self.model(data.cuda())