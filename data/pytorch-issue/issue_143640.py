import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# BEGIN CONFIG #
MODEL_DIR = f'roberta-base'
# END CONFIG #

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, attn_implementation = 'sdpa')
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
input_ids = [tokenizer.encode('Hello world')] * 128
input_ids = torch.stack([torch.tensor(input) for input in input_ids])
attention_mask = torch.ones_like(input_ids)

torch.onnx.dynamo_export(
    model,
    input_ids,
    attention_mask,
    export_options = torch.onnx.ExportOptions(
        dynamic_shapes = True,
        op_level_debug = False,
    )
)

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    f = f'{MODEL_DIR}/model.onnx',
    input_names = ['input_ids', 'attention_mask'], 
    output_names = ['logits'], 
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'}, 
        'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
        'logits': {0: 'batch_size', 1: 'sequence'}
    }, 
    do_constant_folding = True, 
    opset_version = 17,
    dynamo = True
)