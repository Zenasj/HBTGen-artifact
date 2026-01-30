from transformers import AutoModelForCausalLM
import torch

import time


def torch2onnx(model, idx):

    input_ids = torch.randint(
        low=0, high=100, size=(1, input_seq_len[idx]), dtype=torch.int64
    )
    print("start export onnx")
    torch.onnx.export(
        model,
        (input_ids,),
        save_name,
        input_names=["input_ids"],
        output_names=["logits", "past_key_values_cache"],
        verbose=False,
    )

def load(path, idx):
    print(" start load model")
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    torch2onnx(model, idx)
    print("export success.")


model_path = "/home/rzhang/weights/vicuna-7b-v1.5"
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
input_seq_len = [512, 1024, 2048]
save_name = "onnx/model.onnx"

for i in range(3):
    time.sleep(2)

    load(model_path, i)