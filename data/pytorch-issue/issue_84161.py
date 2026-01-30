def prep(x: Any):
    return (x - 128.0)

def forward(self, x):
        x = prep(x)
        ...