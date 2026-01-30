import torch.nn as nn

s_att=nn.Sequential()
s_att.add_module(name="structural_attention_layer",module=structural_attention_layer())
s_att(1,2)