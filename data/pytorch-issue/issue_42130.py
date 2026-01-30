import torch

def div(self, x, y):
    r = torch.div(x, y)
    r = self.activation_post_process(r)
    return r