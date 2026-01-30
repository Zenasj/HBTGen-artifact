from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()
question = "what is google specialization"
text = "Google LLC is an American multinational technology company that specializes in Internet-related services and products, which include online advertising technologies, a search engine, cloud computing, software, and hardware."
encoding = tokenizer.encode_plus(question, text)
input_ids, attention_mask, token_type_ids = encoding["input_ids"],encoding["attention_mask"], encoding["token_type_ids"]

input_ids = torch.tensor([input_ids])
attention_mask = torch.tensor([attention_mask])
token_type_ids = torch.tensor([token_type_ids])

torch.onnx.export(
    model,
    (input_ids,attention_mask, token_type_ids),
    f"{model_name}.onnx",
    input_names = ['input_ids','attention_mask', 'token_type_ids'],
    output_names = ['qa_outputs'], 
    opset_version=12, ##opset has to be set to 12
    do_constant_folding=True,
    use_external_data_format=True,
    dynamic_axes = {
        'input_ids' : {0: 'batch', 1: 'sequence'},
        'attention_mask' : {0: 'batch', 1: 'sequence'}, 
        'token_type_ids' : {0: 'batch', 1: 'sequence'}, 
        'qa_outputs': {0: 'batch'}
    }
)