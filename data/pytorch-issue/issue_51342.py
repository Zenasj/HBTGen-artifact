def forward(self, a: Tensor, shape: List[int]):
      b = a.reshape(shape)
      return b + b