import torch
import time
import torch.multiprocessing as mp


def target(event):
    print('worker sleeping')
    time.sleep(1)
    print('worker: event', event)
    print('worker: target = ', event.query())
    time.sleep(1)
    print('worker: target now = ', event.query())


def main():
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=False, interprocess=True)
    s = torch.cuda.current_stream()

    p = mp.Process(target=target, args=(e0,))
    p.start()

    torch.cuda._sleep(500000000)  # spin for about 500 ms
    s.record_event(e0)
    print('main: ', e0.query())

    p.join()
    print('main:', e0.query())


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()