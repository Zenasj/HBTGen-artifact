import torch

def net_output(self, x, y):
    # Implementation based on self.forward
    xy = torch.cat([x, y], dim=1)  # Concatenate the input data
    uv_pred = self.model(xy)  # Perform prediction using the modified MLP

    p = uv_pred[:,1:2]

'''
omit other irrelevant code
'''

view = p.grad.view(-1)

tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

tensor([[2],
        [5],
        [8]])