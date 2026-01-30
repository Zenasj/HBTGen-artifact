import torch
import torch_xla.core.xla_model as xm
import transformers
import os

os.environ['GPU_NUM_DEVICES'] = '1'
device = xm.xla_device()
model = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()
model = torch.compile(model, backend='aot_torchxla_trace_once')

for batch_size in [1,2,4]:
    for seq_len in [64,128,256,512]:
        print(f'batch_size: {batch_size}, seq_len: {seq_len}')
        input_dict = {
            'input_ids':
                torch.ones((batch_size, seq_len)).to(torch.int64).to(device),
            'attention_mask':
                torch.ones((batch_size, seq_len)).to(torch.int64).to(device),
        }
        with torch.no_grad():
            outputs = model(**input_dict)
            xm.mark_step()

print('Done')