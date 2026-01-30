from transformers import BertForMaskedLM, BertTokenizer
import torch

model = BertForMaskedLM.from_pretrained('./data/Transformer-Bert', torchscript=True)
tokenizer = BertTokenizer.from_pretrained('./data/Transformer-Bert')
model.eval()
model.cuda()

with torch.no_grad():
    inp = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']
    tokenizer_text = [tokenizer.tokenize(i) for i in inp]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
    input_ids = torch.LongTensor(input_ids)

    script_model = torch.jit.trace(model, input_ids.cuda())
    torch.jit.save(script_model, './data/Transformer-Bert/script_model.pt')

from transformers import BertForMaskedLM, BertTokenizer
import torch

# Add following codes, will be ok.
"""
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)
"""

tokenizer = BertTokenizer.from_pretrained('./data/Transformer-Bert')

with torch.no_grad():
    inp = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']
    tokenizer_text = [tokenizer.tokenize(i) for i in inp]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
    input_ids = torch.LongTensor(input_ids)

model = torch.jit.load('./data/Transformer-Bert/script_model.pt')
model.eval()
out = model(input_ids.cuda())
print(out[0].shape)
out = model(input_ids.cuda())
print(out[0].shape)
out = model(input_ids.cuda())
print(out[0].shape)

script_model = torch.jit.trace(model, input_ids.cuda().clone())

model(input_ids.cuda().clone())