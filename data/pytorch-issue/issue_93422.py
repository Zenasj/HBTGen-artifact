from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
text = "Test sentence for transformers bert model."
encoded_input = tokenizer(text, return_tensors="pt")

print(encoded_input)
args = (
    encoded_input["input_ids"],
    encoded_input["attention_mask"],
    encoded_input["token_type_ids"],
)


output = model(**encoded_input)  # this works
output = model(*args)  # this works


from torch import _dynamo as dynamo

fx_module, _ = dynamo.export(model, *args)  # this fails
fx_module, _ = dynamo.export(model, **encoded_input)  # this fails
fx_module.print_readable()