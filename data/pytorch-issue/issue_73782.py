import os
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IWSLT2017
train_iter = IWSLT2017(split='train',language_pair=('en', 'de'))