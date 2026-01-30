from transformers import LlamaForCausalLM
import torch

llm_name = "meta-llama/Llama-2-7b-hf"
llm = LlamaForCausalLM.from_pretrained(llm_name).cuda()

llm.forward = torch.compile(llm.forward, fullgraph=True)

attn_mask = torch.ones(1, 450, dtype = torch.long).cuda()
outputs = llm(torch.ones(1, 450, dtype = torch.long).cuda(), attention_mask=attn_mask)

print("------")

attn_mask = torch.ones(1, 1, dtype = torch.long).cuda()
outputs = llm(torch.ones(1, 1, dtype = torch.long).cuda(), attention_mask=attn_mask)