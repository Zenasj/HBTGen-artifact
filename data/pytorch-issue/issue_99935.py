def forward(self, x):
    x = self.conv(x)
    return self.dropout(x)