def forward(self, x, y):
    # x: [s0, s1]; y: [s2]
    return x.flatten() + y