import torch

py
rowwise_scales = t.abs().amax(dim=[-1])
tensorwise_scales = t.abs().amax()  # could be replaced by rowwise_scales.amax()

py
@torch.compile(fullgraph=True)
def max_on_two_dims(t, first_along_dim, then_along_dim):
    a = t.amax(dim=[first_along_dim], keepdim=True)
    b = t.amax(dim=[first_along_dim, then_along_dim], keepdim=True)
    return a, b

max_on_two_dims(torch.randn((12, 34, 56), device="cuda"), first_along_dim=1, then_along_dim=2)

py
buf0 = empty_strided_cuda((12, 1, 56), (56, 56, 1), torch.float32)
triton_per_fused_amax_0.run(arg0_1, buf0, 672, 34, grid=grid(672), stream=stream0)
buf1 = empty_strided_cuda((12, 1, 1), (1, 1, 1), torch.float32)
triton_red_fused_amax_1.run(arg0_1, buf1, 12, 1904, grid=grid(12), stream=stream0)
del arg0_1

py
@torch.compile(fullgraph=True)
def max_on_two_dims(t, first_along_dim, then_along_dim):
    a = t.amax(dim=[first_along_dim], keepdim=True)
    b = a.amax(dim=[first_along_dim, then_along_dim], keepdim=True)
    return a, b

py
buf0 = empty_strided_cuda((12, 1, 56), (56, 56, 1), torch.float32)
triton_per_fused_amax_0.run(arg0_1, buf0, 672, 34, grid=grid(672), stream=stream0)
del arg0_1
buf1 = empty_strided_cuda((12, 1, 1), (1, 1, 1), torch.float32)
triton_per_fused_amax_1.run(buf0, buf1, 12, 56, grid=grid(12), stream=stream0)

py
def reuse_fwd_amaxes_in_bwd(graph: torch.fx.Graph) -> None:
    abses_by_input = {}
    for abs_ in graph.find_nodes(op="call_function", target=torch.ops.aten.abs.default):
        abses_by_input.setdefault(abs_.args[0], []).append(abs_)

    for source, all_abses in abses_by_input.items():
        abs_ = all_abses[0]
        for other_abs in all_abses[1:]:
            other_abs.replace_all_uses_with(abs_)

        ndims = abs_.meta["val"].ndim

        dims_and_amaxes = []
        for node in abs_.users:
            if node.op == "call_function" and node.target == torch.ops.aten.amax.default:
                assert node.args[0] == abs_
                dims = node.args[1]
                assert isinstance(dims, list)
                keepdim = False if len(node.args) < 3 else node.args[2]
                assert isinstance(keepdim, bool)
                if not keepdim:
                    with graph.inserting_after(node):
                        amax = graph.call_function(torch.ops.aten.amax.default, (abs_, dims, True))
                        squeeze = graph.call_function(torch.ops.aten.squeeze.dims, (amax, dims))
                    node.replace_all_uses_with(squeeze, propagate_meta=True)
                    node = amax
                if not dims:
                    dims = list(range(ndims))
                dims = [d + ndims if d < 0 else d for d in dims]
                dims_and_amaxes.append((sorted(dims), node))

        if not dims_and_amaxes:
            continue

        dims_and_amaxes.sort(key=lambda x: len(x[0]))

        for i, (target_dims, old_amax) in enumerate(dims_and_amaxes):
            for source_dims, source_amax in reversed(dims_and_amaxes[:i]):
                if source_dims == target_dims:
                    old_amax.replace_all_uses_with(source_amax)
                    dims_and_amaxes[i] = (target_dims, source_amax)
                    break
                if all(d in target_dims for d in source_dims):
                    remaining_dims = [d for d in target_dims if d not in source_dims]
                    with graph.inserting_after(source_amax):
                        new_amax = graph.call_function(torch.ops.aten.amax.default, (source_amax, remaining_dims, True))
                    old_amax.replace_all_uses_with(new_amax, propagate_meta=True)
                    dims_and_amaxes[i] = (target_dims, new_amax)
                    break

        biggest_amax = dims_and_amaxes[-1][1]

        for node in abs_.users.copy():
            if node.op == "call_function" and node.target == torch.ops.aten.max.default and node.args == (abs_,) and node.kwargs == {}:
                node.replace_input_with(abs_, biggest_amax)

    graph.eliminate_dead_code()