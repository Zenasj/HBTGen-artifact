# test.py

import torch
import onnx
from transformers import BertConfig, BertModel


def main():
    model = BertModel(BertConfig(num_hidden_layers=1))
    model.eval()

    input_ids = torch.randint(30522, (1, 512), dtype=torch.long)

    onnx_model_path = 'net.onnx'
    torch.onnx.export(model, input_ids, onnx_model_path, export_params=True,
                      opset_version=13, verbose=False, input_names=['input'],
                      output_names=['output1', 'output2'],
                      )

    model = onnx.load(onnx_model_path)
    weights = model.graph.initializer
    for w in weights:
        print(w.name, w.dims)


main()