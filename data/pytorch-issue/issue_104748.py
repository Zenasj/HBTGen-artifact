import torch
from transformers import BertModel

# torch.randint(0, 30522, (1, 128), dtype=torch.long, device='cuda')  # Input shape (batch, sequence_length)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids):
        return self.bert(input_ids)[0]  # Return last_hidden_state for compatibility with ONNX export

def my_model_function():
    model = MyModel()
    model = model.to('cuda')  # Match original code's device placement
    return model

def GetInput():
    # Generate input_ids matching BERT's vocabulary size (30522) and typical sequence length
    return torch.randint(0, 30522, (1, 128), dtype=torch.long, device='cuda')

