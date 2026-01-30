import torch.onnx
from transformers import BertTokenizer, BertModel
import torch
import onnx
import onnxruntime
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Get the first example data to run the model and export it to ONNX
max_seq_length = 512
inputs = {"input_ids": torch.ones(1, max_seq_length, dtype=torch.int64),
          "attention_mask": torch.ones(1, max_seq_length, dtype=torch.int64),
          "token_type_ids": torch.ones(1, max_seq_length, dtype=torch.int64)}

outputs = model(**inputs)

# export model to onx
symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
torch.onnx.export(model, args=tuple(inputs.values()), f='bert_onnx.onnx',
                  input_names=['input_ids','token_type_ids','attention_mask'],
                  output_names=['output'],
                  dynamic_axes={'input_ids': symbolic_names,  # variable lenght axes
                                'token_type_ids': symbolic_names,
                                'attention_mask': symbolic_names},
                  do_constant_folding=True,
                  opset_version=11)
                  #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

onnx_model = onnx.load("bert_onnx.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("bert_onnx.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
# inputs to onnx format
ort_inputs = {
'input_ids':inputs['input_ids'].cpu().numpy(),
"attention_mask":inputs['attention_mask'].cpu().numpy(),
"token_type_ids":inputs['token_type_ids'].cpu().numpy()
}
# outputs
ort_outs = ort_session.run(None, ort_inputs)
print('####ONNX Result')
print(ort_outs[0])

# Sanity check
# compare ONNX Runtime and PyTorch results
# export input
np.testing.assert_allclose(to_numpy(outputs.last_hidden_state), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# new input
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
new_inputs = tokenizer("Hello world", padding='max_length' , return_tensors="pt")
new_outpus = model(**new_inputs)
new_hidden_state = new_outpus.last_hidden_state
print('#### Last hidden state ',new_hidden_state)
new_ort_inputs = {
'input_ids':new_inputs['input_ids'].cpu().numpy(),
"attention_mask":new_inputs['attention_mask'].cpu().numpy(),
"token_type_ids":new_inputs['token_type_ids'].cpu().numpy()
}
new_ort_outs = ort_session.run(None, new_ort_inputs)
print('#### NEW ONNX Result')
print(new_ort_outs[0])
np.testing.assert_allclose(to_numpy(new_hidden_state), new_ort_outs[0], rtol=1e-03, atol=1e-05)