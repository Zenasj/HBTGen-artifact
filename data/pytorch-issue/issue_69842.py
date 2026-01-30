import torch
import torch.nn.functional as F

def forward(self, x: Tensor) -> Tensor:
         if self.is_first_iteration():
             self.state_xt = [torch.empty((x.shape[0], x.shape[1], 0))] * len(self.convs)
             self.state_x = [torch.empty((x.shape[0], x.shape[1], 0))] * len(self.convs)
 
         for idx, c in enumerate(self.convs):
             xt = F.leaky_relu(x, self.LRELU_SLOPE)
             xt = c(xt, self.is_last())
             x = torch.cat([self.state_x[idx], x], dim=-1)
             xt = torch.cat([self.state_xt[idx], xt], dim=-1)