import torch
import torch._dynamo as dynamo

@dynamo.optimize("eager",dynamic=True)
def predict(text):
    tensor = torch.ByteTensor(list(bytes(text, 'utf8')))
    return tensor + tensor

text = "This is a note about Frank Sinatra who was in Saint Genis Pouilly during summer of 75. He was not really there."
magic_no = [99,100]
for i in magic_no:
    input = text[:i]
    predict(input)