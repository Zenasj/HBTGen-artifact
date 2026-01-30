import torch
import time

from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, LoggingTensorMode, \
    log_input, capture_logs, capture_logs_with_logging_tensor_mode

def print_table_header():
    print(f"{'b':<10} {'m':<12} {'mm + copy':<18} {'bmm':<18}")

def print_table_row(b, m, time_grad_true, time_grad_false):
    print(f"{b:<10} {m:<12} {time_grad_true:<18.6f} {time_grad_false:<18.6f}")

def measure(B, M):
    def run(x, y):
        out = x @ y
        torch.cuda.synchronize()
        return out

    # [n, k]
    a = torch.rand([768, 768], dtype=torch.half, device="cuda", requires_grad=True)
    a_detach = a.detach()

    # [b, m, n]
    b = torch.rand([B, M, 768], dtype=torch.half, device="cuda").transpose(1, 2).contiguous()

    def fn(t1, t2):
        return t1 @ t2

#    compiled_fn = torch.compile(fn, mode="max-autotune")
#    compiled_fn(a_detach, b)
#
#    def wrapped(t1, t2):
#       fn(t1, t2)
#       torch.cuda.synchronize()
#
#    # warm
#    for _ in range(1000):
#        wrapped(a_detach, b)
#
#    before = time.perf_counter()
#
#    # test
#    for _ in range(1000):
#        wrapped(a_detach, b)
#    after = time.perf_counter()
#    print(after - before)


    # print("require_grad=True")
    # warm
    for _ in range(1000):
        run(a, b)

    with capture_logs_with_logging_tensor_mode() as logs:
       run(a, b)
    # print("\n".join(logs))

    before = time.perf_counter()

    # test
    for _ in range(1000):
        run(a, b)
    after = time.perf_counter()
    t_req_grad_true = after - before


    # print("require_grad=False")
    # warm
    for _ in range(1000):
        run(a_detach, b)

    with capture_logs_with_logging_tensor_mode() as logs:
       run(a_detach, b)
    # print("\n".join(logs))
    before = time.perf_counter()

    # test
    for _ in range(1000):
        run(a_detach, b)
    after = time.perf_counter()
    t_req_grad_false = after - before

    print_table_row(B, M, float(t_req_grad_true), float(t_req_grad_false))
    # print("results allclose: ", torch.allclose(run(a, b), run(a_detach, b)))
    # print("max rel diff: ", ((run(a, b) -  run(a_detach, b)) / (run(a, b))).abs().max())


print_table_header()
measure(2048, 4)
measure(1024, 8)
measure(512, 64)
measure(128, 128)
measure(64, 512)
measure(8, 1024)
measure(4, 2048)