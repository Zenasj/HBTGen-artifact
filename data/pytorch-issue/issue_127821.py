import torch
from torch._decomp import register_decomposition

lib = torch.library.Library("fsdp_test", "DEF")

lib.define("chunk_cat_(Tensor(a!) ret, Tensor[] tensors, int dim, int num_chunks) -> ()")

@torch.library.impl(lib, "chunk_cat_", "Meta")
def chunk_cat_(ret, tensors, dim, num_chunks):
    torch._chunk_cat(
        tensors, dim, num_chunks, out=ret
    )

@torch.library.impl(lib, "chunk_cat_", "CUDA")
def chunk_cat_(ret, tensors, dim, num_chunks):
    torch._chunk_cat(tensors, dim, num_chunks, out=ret)


def f(x, y, z):
    full_default_3: "f32[2, 524544]" = torch.ops.aten.full.default([2, 524544], 1.0, dtype = torch.float32, layout = torch.strided, device = "cuda", pin_memory = False)
    chunk_cat_default_1 = torch.ops.fsdp_test.chunk_cat_.default(full_default_3, [x, y, z], 0, 2)
    mul_out = torch.mul(full_default_3, full_default_3)
    sum_out = mul_out.sum()
    return sum_out


if __name__ == "__main__":
    x = torch.randn([1024, 512], device="cuda")
    y = torch.randn([512], device="cuda")
    z = torch.randn([1024, 512], device="cuda")
    eager_out = f(x, y, z)

    compiled_aot_eager_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    compiled_aot_eager_out = compiled_aot_eager_f(x, y, z)
    assert torch.allclose(eager_out, compiled_aot_eager_out), f"eager_out: {eager_out}, compiled_aot_eager_out: {compiled_aot_eager_out}"   # passes

    compiled_inductor_f = torch.compile(f, backend="inductor", fullgraph=True)
    compiled_inductor_out = compiled_inductor_f(x, y, z)
    assert torch.allclose(eager_out, compiled_inductor_out), f"eager_out: {eager_out}, compiled_inductor_out: {compiled_inductor_out}"  # fails

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 512), (512, 1))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (1024, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2, 524544), (524544, 1), torch.float32)
        # Source Nodes: [full_default_3], Original ATen: [aten.full]
        stream0 = get_raw_stream(0)
        triton_poi_fused_full_0.run(buf0, 1049088, grid=grid(1049088), stream=stream0)
        # Source Nodes: [full_default_3], Original ATen: [aten.full]
        buf1 = torch.ops.fsdp_test.chunk_cat_.default(buf0, [arg0_1, arg1_1, arg2_1], 0, 2)
        del arg0_1
        del arg1_1
        del arg2_1
        buf3 = empty_strided_cuda((129, ), (1, ), torch.float32)
        # Source Nodes: [full_default_3, mul_out, sum_out], Original ATen: [aten.full, aten.mul, aten.sum]
        triton_red_fused_full_mul_sum_1.run(buf3, 129, 8133, grid=grid(129), stream=stream0)
        del buf0
        buf4 = empty_strided_cuda((), (), torch.float32)
        # Source Nodes: [full_default_3, mul_out, sum_out], Original ATen: [aten.full, aten.mul, aten.sum]
        triton_per_fused_full_mul_sum_2.run(buf3, buf4, 1, 129, grid=grid(1), stream=stream0)
        del buf3
    return (buf4, )