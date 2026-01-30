import torch

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(interprocess=True) # what I want to share between the streams by ipc_handle
end_event_ipc_handle = end_event.ipc_handle()
pin1_event = torch.cuda.Event(enable_timing=True)
pin2_event = torch.cuda.Event(enable_timing=True)

with torch.cuda.stream(torch.cuda.Stream()):
    start_event.record()
    
    # Run some things here
    
    pin1_event.record()
    end_event.record()

with torch.cuda.stream(torch.cuda.Stream()):
    end_event = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), end_event_ipc_handle)
    end_event.wait() # wait asynchronously

    # Run some things here

    pin2_event.record()

torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(pin1_event)
print(f"Elapsed time: {elapsed_time_ms} ms")

elapsed_time_ms = pin1_event.elapsed_time(pin2_event)
print(f"Elapsed time: {elapsed_time_ms} ms")

ctx = mp.get_context('spawn')
ctx.SimpleQueue()

import time
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process

SIZE = (10000, 10000)

def stream1(queue): 
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    interproc_event = torch.cuda.Event(interprocess=True)
    interproc_event_ipc_handle = interproc_event.ipc_handle()
    queue.put(interproc_event_ipc_handle)
    
    with torch.cuda.stream(torch.cuda.Stream()):
        print("stream1: ", torch.cuda.current_stream())

        start_event.record()
        
        # Perform matrix multiplication
        for i in range(100):
            matrix1 = torch.rand(SIZE, device='cuda')
            matrix2 = torch.rand(SIZE, device='cuda')
            _ = torch.matmul(matrix1, matrix2)

        interproc_event.record()
        end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"stream1's elapsed time: {elapsed_time_ms} ms")
    
    
def stream2(queue):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    interproc_event_ipc_handle = queue.get()
    interproc_event = torch.cuda.Event.from_ipc_handle(torch.cuda.current_device(), interproc_event_ipc_handle)
    
    with torch.cuda.stream(torch.cuda.Stream()):
        print("stream2: ", torch.cuda.current_stream())

        interproc_event.wait()

        start_event.record()
        
        # Perform matrix multiplication
        for i in range(100):
            matrix1 = torch.rand(SIZE, device='cuda')
            matrix2 = torch.rand(SIZE, device='cuda')
            _ = torch.matmul(matrix1, matrix2)

        end_event.record()
    torch.cuda.synchronize()

    time.sleep(1) # avoid nested printing

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"stream2's elapsed time: {elapsed_time_ms} ms")


def main():
    ctx = mp.get_context('spawn')
    queue = ctx.SimpleQueue()

    p1 = Process(target=stream1, args=(queue,))
    p2 = Process(target=stream2, args=(queue,))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()
    
    
if __name__ == '__main__':
    main()