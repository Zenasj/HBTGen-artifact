import transformers
import torch._dynamo
torch._dynamo.config.suppress_errors = False
    
def make_data(model, device):
    batch_size = 1
    seq_len = 16
    input = torch.randint(
        low=0, high=model.config.vocab_size, size=(batch_size, seq_len), device=device
    )

    label = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, seq_len),
                          device=device)
    return input, label

device = torch.device('cuda')
config = transformers.AutoConfig.from_pretrained("facebook/opt-125m")
config.tie_word_embeddings = False
model = transformers.OPTForCausalLM(config=config)
model.to(device)

optimized_model = torch.compile(model, backend='inductor',options={'trace.enabled':True,'trace.graph_diagram':True})
data = make_data(model, device)
model.zero_grad(set_to_none=True)
with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
    torch._dynamo.explain(model, data[0])