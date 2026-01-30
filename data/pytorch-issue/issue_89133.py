import torch

def test_cpu_gpu_parity(self):
    cpu_model = TransformerWithSharedParams.init(
        self.process_group,
        FSDPInitMode.NO_FSDP,
        CUDAInitMode.CUDA_NEVER,
        deterministic=True,
    )
    gpu_model = copy.deepcopy(cpu_model).cuda()
    cpu_inp = cpu_model.get_input(torch.device("cpu"))
    gpu_inp = gpu_model.get_input(torch.device("cuda"))
    for t1, t2 in zip(cpu_inp, gpu_inp):
        assert torch.equal(t1, t2.cpu())  # same input except device
    for p1, p2 in zip(cpu_model.parameters(), gpu_model.parameters()):
        assert torch.equal(p1, p2.cpu())  # same parameters except device
    cpu_out = cpu_model(*cpu_inp)
    cpu_out.sum().backward()
    gpu_out = gpu_model(*gpu_inp)
    gpu_out.sum().backward()
    for p1, p2 in zip(cpu_model.parameters(), gpu_model.parameters()):
        assert torch.equal(p1, p2.cpu())
        assert torch.equal(p1.grad, p2.grad.cpu()), f"{torch.linalg.vector_norm(p1.grad - p2.grad.cpu())}"

assert torch.equal(p1.grad, p2.grad.cpu()), f"{torch.linalg.vector_norm(p1.grad - p2.grad.cpu())}"
AssertionError: 2.398389005975332e-05