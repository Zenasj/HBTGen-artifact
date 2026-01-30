def __contains__(self, val):
    return (val == self).any().item()