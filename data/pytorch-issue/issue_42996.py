import torch

@autocast()
def forward(self, prev, prior):
        # type: (List[Tensor], Tensor) -> Tensor
        i_vals = torch.mm(prev[0], self.weight_ih_0.t()) \
                  + torch.mm(prev[1], self.weight_ih_1.t()) \
                  + torch.mm(prev[2], self.weight_ih_2.t()) \
                  + self.bias_ih
        h_vals = torch.mm(prior, self.weight_hh.t()) + self.bias_hh

       # this is for debugging
        print("i_vals type", i_vals.type())
        print("h_vals type", h_vals.type())
        print("h_vals half type", h_vals.half().type())

        r_i, z_i, n_i = i_vals.chunk(3, 1)
        r_h, z_h, n_h = h_vals.chunk(3, 1)
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        new = torch.tanh(n_i + (r * n_h))
        return new * z + prior * (1-z)