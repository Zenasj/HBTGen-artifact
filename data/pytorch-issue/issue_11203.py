AT_CHECK(
        N + M <= size(),
        "ArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        size());

N + M <= size()