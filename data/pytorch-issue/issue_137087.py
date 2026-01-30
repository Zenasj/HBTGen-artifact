import torch

def test_mark_unbacked_slice(self):
        @torch.compile(backend="inductor", mode="reduce-overhead", fullgraph=True)
        def f(x):
            return x.sum()
            
        x = torch.empty_strided((1, 4), (5, 1), device="cuda")
        torch._dynamo.decorators.mark_unbacked(x, 0)
        f(x)

def complex_memory_overlap(t: torch.Tensor) -> bool:
    # if torch._debug_has_internal_overlap thinks this tensor potentially has
    # memory overlap internally, let's dig deeper to find out whether it's true.
    #               
    # Call squeeze() so that dimension with size 1 does not cause false positive.
    t = index_expanded_dims(t, get_expanded_dims(t)).squeeze()
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        indices = [x for _, x in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            if strides[indices[i]] < prev_stride * prev_size:
                return True 
    return False