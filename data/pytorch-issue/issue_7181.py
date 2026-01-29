import torch
import torch.multiprocessing as mp
import os

# torch.rand(2, 1000, dtype=torch.float32)  # Input shape inferred from the kernel's batch_tensor
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy model to fulfill structure requirements
        self.identity = torch.nn.Identity()

    def forward(self, x):
        return self.identity(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 1000, dtype=torch.float32)

def worker(queue, exit_queue, device):
    model = my_model_function()
    input_tensor = GetInput()
    output = model(input_tensor)
    queue.put(output)
    print(f"Worker {os.getpid()} finished processing")
    exit_queue.get()  # Wait for exit signal

def run_synchronized_multiprocessing():
    mp.set_start_method("spawn")
    exit_queue = mp.Queue()
    q = mp.Queue()
    processes = []
    device_list = [0, 1]
    for device in device_list:
        p = mp.Process(target=worker, args=(q, exit_queue, device))
        p.start()
        processes.append(p)
    
    # Read all results first before signaling workers to exit
    results = [q.get() for _ in range(len(device_list))]
    
    # Send exit signals to all workers
    for _ in range(len(device_list)):
        exit_queue.put(None)
    
    for p in processes:
        p.join()
    
    return results

