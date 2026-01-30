import torch
import torch.multiprocessing as mp

@torch.compile
def affine(x, a, b):
    return a * x + b

def run_fn(recv_q):
    while True:
        x_recv = recv_q.get()
        if x_recv is None:
            break
        x = torch.empty_like(x_recv)
        x.copy_(x_recv)
        del x_recv
        _ = affine(x, 3.0, 2.0)

def main():
    ctx = mp.get_context("spawn")
    send_q = ctx.SimpleQueue()
    proc = ctx.Process(
        target=run_fn,
        args=(send_q,),
        daemon=True,
    )
    proc.start()

    # Separate scope to not hold onto tensors
    def test():
        for _ in range(2):
            x = torch.randn(100, 100, device="cuda")
            send_q.put(x)

    test()
    send_q.put(None)
    print("Waiting on subprocess...")
    proc.join()
    print("Done.")

if __name__ == "__main__":
    main()