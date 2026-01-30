import torch
import torch.nn as nn

def save_ip_adapter_weight(self,weight_save_path):
        assert self.use_ip_adapter
        filtered_attn_processors={k:self.unet.attn_processors[k] for k in self.unet.attn_processors if isinstance(self.unet.attn_processors[k],torch.nn.Module)}
        ip_layers=torch.nn.ModuleList(filtered_attn_processors.values())
        torch.save(ip_layers.state_dict(),weight_save_path)

state_dict_save_during_training=torch.load('./ip_adapter_weight.bin') # load weight saved during training
state_dict_to_cpu={k:state_dict_save_during_training[k].cpu() for k in state_dict_save_during_training.keys()} # move to cpu
torch.save(state_dict_to_cpu,'./temp_state_dict_to_cpu.bin') # save cpu weight

state_dict_save_during_training=torch.load('./ip_adapter_weight.bin')
state_dict_to_cpu=torch.load('./temp_state_dict_to_cpu.bin')
assert len(state_dict_save_during_training.keys())==len(state_dict_to_cpu.keys())
for k1,k2 in zip(state_dict_save_during_training.keys(),state_dict_to_cpu.keys()):
    assert k1==k2
    
    assert torch.equal(state_dict_save_during_training[k1],state_dict_to_cpu[k2].cuda())
    assert torch.equal(state_dict_save_during_training[k1].cpu(),state_dict_to_cpu[k2])
    
    assert state_dict_save_during_training[k1].grad is None
    assert state_dict_to_cpu[k2].grad is None
    
    assert state_dict_save_during_training[k1].requires_grad is False
    assert state_dict_to_cpu[k2].requires_grad is False
    
    assert state_dict_save_during_training[k1].dtype is torch.float16
    assert state_dict_to_cpu[k2].dtype is torch.float16

for k, v in state_dict_save_during_training.items():
    torch.save(v, f'./gpu_{k}.bin')
    torch.save(v.cpu(), f'./cpu_{k}.bin')

state_dict_save_during_training=torch.load('./ip_adapter_weight.bin') # load weight saved during training
state_dict_to_cpu={k:state_dict_save_during_training[k].clone() for k in state_dict_save_during_training.keys()} # clone
torch.save(state_dict_to_cpu,'./temp_state_dict_cloned.bin') # save cloned weight

3
x = torch.ones(1000**3)
y = x[:5]  # y is a view of the first 5 entries of x