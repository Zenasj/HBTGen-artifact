import multiprocessing as mp
import faulthandler

# Just to print the segfault in the child
faulthandler.enable()

def _mp_fn(barrier):
    barrier.wait()

if __name__ == '__main__':
    barrier = mp.get_context("fork").Barrier(1)
    p = mp.get_context("spawn").Process(target=_mp_fn, args=(barrier,))
    p.start()
    p.join()