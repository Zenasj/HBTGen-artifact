import torch.nn.functional as F

from dataclasses import dataclass
import math 
import torch 
import torch.nn as nn 
from torch.nn import functional as F

#Write the GPT Model 


class CausalSelfAttention(nn.Module):
    # Multiple heads that function in parallel
    # Concatenate their outputs

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        #output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        #regularisation 
        self.n_head = config.n_head
        self.n_embd = config.n_embd 

        #If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
        #torch.tril returns the lower triangular area of a matrix
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1,1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() #Batch Size, Sequence Length, Embedding Dimensionality
        ## This part doesn't fully make sense to me because of dim mismatch
        qkv = self.c_attn(x)  
        q,k,v = qkv.split(self.n_embd, dim=2)

        #nh is "number of heads", hs is "head size"
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y        


class MLP (nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)

        #Attention is a weighted sum operation, aggregation function (where they communicate)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        #MLP is the map (where they think individually about what happens)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

@dataclass
class GPTConfig: 
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #no of tokens (50,000 BPE merges, 256 byte tokens, 1 <|endoftext|>)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #nn.Embedding is a glorified wrapper around a tensor that allows you to index into its rows
            wpe = nn.Embedding(config.block_size, config.n_embd),
            #ModuleList allows you to index into it with integers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )) 
        # Final classifier - projects the embedding size to the vocab size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    def forward(self, idx, targets=None):
        #To actually be able to generate from this model, we have to forward it 
        #idx is of shape (B, T) - token indices
        #B independent sequences, each of length t
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is length {self.config.block_size}"
        #forward the token and positional embeddings
        #arange is a version of range for pytorch
        pos = torch.arange(0, T, dtype=torch.long, device = idx.device) #shape T just a long string
        pos_emb = self.transformer.wpe(pos) #position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)
        #position embeddings are constant for every row of input. So there is broadcasting. 
        x = tok_emb + pos_emb

        #forward transformer blocks 
        for block in self.transformer.h:
            x = block(x)
        
        #forward the final layernorm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        loss=None
        if targets is not None:
            # Flattening 3d tensor into 2d (B*T, vocab size where each row represents token)
            # Flatten targets into a single tensor of B*T
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from hugging-face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # Create a from scratch initialised minGPT model
        config = GPTConfig(**config_args) 
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Initialise a HF/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        #tf transposes these weights from what pytorch would want ... tf? 
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else: 
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ----------------------------------------------------------------------------------------------------------------

num_return_sequences = 5
max_length = 30 
device = torch.device('mps')


#get a data batch
import tiktoken 
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T +1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

#get logits
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward() #accumulates the gradient from loss
    optimizer.step() #updates the parameters
    print(f"step {i}, loss: {loss.item()}") #.item converts a tensor (on gpu) which is shipped back to cpu as a single float 



print(logits.shape)
print(loss)