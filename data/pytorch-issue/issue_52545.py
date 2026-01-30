if self.out_proj.bias is not None:
    constant_(self.out_proj.bias, 0.)