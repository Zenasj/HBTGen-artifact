python
def encode(name):
    def hook(var):
        return {'name': name, 'grad': var.grad}
    return hook
def decode(hook_output):
    return hook_output['grad']

for name, param in model.named_paramers():
    param.register_hook(encode(name))

# later; psuedo-code
for hook_out in hook_outputs:
    grad = decode(hook_output)

python
def encode(var):
    print('In encode')
    return {'grad': var.grad, 'other': stuff}
def decode(encode_output):
    print('In decode')
    return encode_output['grad']

# in setup
for param in model.named_parameters():
    param.encoding(encode)
    param.decoding(decode)

# later in training
loss.backwards() 
# "In encode" (prints at least once)
optimizer.step() 
# "In decode" (prints at least once)