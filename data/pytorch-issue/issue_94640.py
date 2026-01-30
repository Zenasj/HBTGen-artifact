python
def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
        size = [1] * ref.dim()
        size[dim] = -1
        return src.view(size).expand_as(ref)

def scatter(src: Tensor, index: Tensor, dim: int = 0,
                dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:

    dim = src.dim() + dim if dim < 0 else dim

    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    size = list(src.size())
    size[dim] = dim_size

    count = src.new_zeros(dim_size)
    count.scatter_add_(0, index, src.new_ones(src.size(dim)))
    count = count.clamp_(min=1)

    index = broadcast(index, src, dim)
    out = src.new_zeros(size).scatter_add_(dim, index, src)

    return out / broadcast(count, out, dim)