for layer in self.layers.values():
    h = layer(h, self.freqs_cis)