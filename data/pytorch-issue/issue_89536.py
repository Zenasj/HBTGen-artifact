def forward(self, a, b, c):
        """
        The XLA graph will only return the first 2 items
        """
        return a + b, a + c, b

def forward(self, a, b, c):
        """
        Inplace update on b cause it to be returned in XLA graph
        """
        b.zero_()
        return a + b, a + c, b

def forward(self, a, b, c):
        """
        Even if we return b twice, the XLA graph only return b once.
        """
        b.zero_()
        return a + b, a + c, b, b