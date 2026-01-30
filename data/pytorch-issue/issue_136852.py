outs_with_identical_metadata_that_require_grad = (
                []
                if not isinstance(o, Tensor)
                else [
                    curr
                    for curr in out_storage_to_tensors[curr_storage]
                    if has_same_metadata(o, curr)
                    and curr.requires_grad
                    and o is not curr
                ]
            )