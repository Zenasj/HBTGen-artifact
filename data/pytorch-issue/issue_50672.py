def forward(self, x):
    return self.pe[:, :x.size(1)]

def forward(self, x):
    size = x.size(1);  x = None
    pe = self.pe
    getitem = pe.__getitem__((slice(None, None, None), slice(None, size, None)));  pe = size = None
    return getitem