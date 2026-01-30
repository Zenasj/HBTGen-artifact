import torch

input_ids = [2, 184, 20, 745, 2667, 11318, 18, 37, 4335, 4914,3]
token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

input_ids_tensor = torch.tensor([input_ids])
token_type_ids_tensor = torch.tensor([token_type_ids])
attention_mask_tensor = torch.tensor([attention_mask])

cur_state_dict = torch.load("./albert_model/checkpoint-69203/pytorch_model.bin", map_location="cpu")

model = AutoModelForSequenceClassification.from_pretrained("./albert_model/checkpoint-69203")

model.load_state_dict(cur_state_dict, strict=False)
model.eval()

with torch.no_grad():
    output = model(input_ids_tensor, token_type_ids_tensor, attention_mask_tensor)
    print("output: ", output)

    torch.onnx.export(model, 
                    (input_ids_tensor, token_type_ids, attention_mask), 
                    "new_onnx_model", 
                    export_params=True, 
                    opset_version=16,
                    do_constant_folding=True,
                    input_names = ["input_ids", "token_type_ids", "attention_mask"],
                    output_names = ["logits", "scores"])