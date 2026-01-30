@parse_args('v', 'v', 'is')
def max_unpool2d(g, self, indices, output_size):
    return g.op("max_unpool2d", self, indices, output_size)