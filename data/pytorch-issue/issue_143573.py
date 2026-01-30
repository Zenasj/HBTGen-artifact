def forward_pre_hook(module, input, output=None):
    print_rank_0("forward_pre_hook")
def forward_hook(module, input, output=None):
    print_rank_0("forward_hook")
def backward_pre_hook(module, input, output=None):
    print_rank_0("backward_pre_hook")
def backward_hook(module, input, output=None):
    print_rank_0("backward_hook")
for name, module in unet.named_modules():
    if name == "module.up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2":
        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)