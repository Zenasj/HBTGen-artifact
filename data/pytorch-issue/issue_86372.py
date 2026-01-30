@register_decomposition([prims.slice_in_dim])
def slice_in_dim(a, start_index, limit_index, stride=1, axis=0):
    return aten.slice(a, dim=axis, start=start_index, end=limit_index, step=stride)