import torch

model = AutoModelForCausalLM.from_pretrained(
        'llama-7b-path',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
model = model.eval()
example_datas = (torch.LongTensor([ 1,   529, 29989,  1792, 29989, 29958,    13, 30287,   233,   175]).cuda(), )
exported_model = capture_pre_autograd_graph(model, example_datas)

model(*example_datas)