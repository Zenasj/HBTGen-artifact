import torch
import torch._dynamo

from transformers.tokenization_utils_base import BatchEncoding


@torch._dynamo.optimize('eager')
def tokenization(x):
    encoding = BatchEncoding({'key': x})
    return encoding['key']


x = torch.rand((1, 4))
y = tokenization(x)