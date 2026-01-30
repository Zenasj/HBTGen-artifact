def __init__(self, consensus_type, dim=1):
    self.consensus_type = consensus_type
    self.dim = dim
    self.shape = None

def forward(self, input_tensor):
    self.shape = input_tensor.size()
    if self.consensus_type == 'avg':
        output = input_tensor.mean(dim=self.dim, keepdim=True)
    elif self.consensus_type == 'identity':
        output = input_tensor
    else:
        output = None

    return output

def backward(self, grad_output):
    if self.consensus_type == 'avg':
        grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
    elif self.consensus_type == 'identity':
        grad_in = grad_output
    else:
        grad_in = None

    return grad_in