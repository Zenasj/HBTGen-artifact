def forward(self, x):
        y = x + 1
        z = y.to("cpu")
        z.add_(5)
        return x

def forward(self, x):
        y = x + 1
        z = y.to("cpu")
        z.add_(5)
        return z