def _create_block_mask_inner(
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str,
    KV_BLOCK_SIZE: int,
    Q_BLOCK_SIZE: int,
):
    r"""Work around for being unable to instantiate __torch_function__ mode under compile.
    `create_block_mask` will compile this inner function and wrap the call to this
    with the __torch_function__ mode.
    """
    mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, _compile=True)
    full_block_mask, partial_block_mask = _convert_mask_to_block_mask(
        mask_tensor,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        separate_full_blocks=True,
    )
    return _create_sparse_block_from_block_mask(
        (full_block_mask, partial_block_mask), mask_mod
    )