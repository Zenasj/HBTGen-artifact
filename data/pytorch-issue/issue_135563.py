import torch

model_1 = torch._export.aot_load('./' + so_path, "cuda")
with torch.inference_mode():
    output = model_1(test_batch)
print('done predict with loaded model')

[tasklist]
### Tasks