def upsample_nearestnd(
    x,
    output_size,
    scales_x: Tuple[Optional[float], ...],
    n: int = 2,
    exact: bool = False,
):
   # ...
    scales = [i / o for i, o in zip(i_sizes, o_sizes)]
    for i, scale in enumerate(scales):
        if scale:
            scales[i] = scale

def upsample_nearestnd(
    x,
    output_size,
    scales_x: Tuple[Optional[float], ...],
    n: int = 2,
    exact: bool = False,
):
   # ...
    scales = [i / o for i, o in zip(i_sizes, o_sizes)]
    for i, scale in enumerate(scales_x):
        if scale:
            scales[i] = scale