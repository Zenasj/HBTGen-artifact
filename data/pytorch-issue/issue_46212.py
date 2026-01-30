import torch.nn as nn

import torch
from typing import List


class Token():
    def __init__(self, text: str):
        self.text = text


class MyTokenizer(torch.nn.Module):
    def __init__(self):
        super(MyTokenizer, self).__init__()

    def forward(self, sentence: str):
        """
        Create list of Tokens by splitting string on spaces.
        """
        result: List[Token] = []
        tokenized = sentence.split(' ')
        for word in tokenized:
            result.append(Token(word))
        return result


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, tokens: List[Token]):
        """
        Concatenate Tokens to string.
        """
        result: str = ""
        for token in tokens:
            result += token.text
        return result


test_sample = "this is a test"

my_tokenizer = MyTokenizer()
jit_my_tokenizer = torch.jit.script(my_tokenizer)
tokens = jit_my_tokenizer(test_sample)
print(tokens)
jit_my_tokenizer.save("jit_my_tokenizer.pt")

model = MyModel()
jit_model = torch.jit.script(model)
outputs = jit_model(tokens)
print(outputs)
jit_model.save("jit_model.pt")