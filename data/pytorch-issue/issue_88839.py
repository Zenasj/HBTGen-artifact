import time
import torch
import torch.multiprocessing as mp

def set_device():
    # Note: the code can run if the following two lines are commented out
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return

def worker(job_queue: mp.Queue, done_queue: mp.Queue, result_queue: mp.Queue):
    set_device()
    para = torch.zeros((100, 100))
    try:
        while True:
            result = para + torch.randn_like(para)

            if not job_queue.empty():
                job_queue.get()
                break
            if result_queue.full():
                time.sleep(0.1)
                continue
            result_queue.put(result)

        done_queue.put(None)
        result_queue.cancel_join_thread()

    except Exception as e:
        print(f'{mp.current_process().name} - {e}')

def test_queue():
    set_device()
    ctx = mp.get_context('spawn')
    job_queue = ctx.Queue()
    result_queue = ctx.Queue(100)
    done_queue = ctx.Queue()
    proc = ctx.Process(target=worker, args=(job_queue, done_queue, result_queue))
    proc.start()
    for i in range(10):
        result = result_queue.get()
        for j in range(100):
            if not result_queue.empty():
                result = result_queue.get()
            else:
                break
        print("result: ", result.sum().item())
        time.sleep(0.1)
    job_queue.put(None)
    proc.join()

if __name__ == '__main__':
    test_queue()