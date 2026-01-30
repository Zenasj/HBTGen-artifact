input = input.contiguous().reshape({n, c, 1, -1});
target = target.contiguous().reshape({n, 1, -1});

input = input.contiguous().view({n, c, 1, -1});
target = target.contiguous().view({n, 1, -1});