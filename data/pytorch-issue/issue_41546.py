import torch

torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None)
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None)
torch.distributed.rpc.remote(to, func, args=None, kwargs=None)

def process1():
    rpc.init_rpc(...)
    some_signal = Switch(False)           # a switch class
    rpc.pair("signal", some_signal)   # expose this variable in the implicit global rpc group
    sleep(10)                              # do some initialization
    some_signal.set(True)
    # synchronized
   
def process2():
    rpc.init_rpc(...)
    some_signal = Switch(False)           # a switch class
    rpc.pair("signal2", some_signal) # expose this variable in the implicit global rpc group
    sleep(5)                              # do some initialization
    some_signal.set(True)
    while not rpc.get_paired("signal"):
        sleep(1)
    # synchronized

def process2():
    rpc.init_rpc(...)
    while not rpc.get_paired("signal") or not rpc.get_paired("signal2"):
        sleep(1)
    # synchronized

signal1 = False
signal2 = False

def get_signal1():
    return signal1

def get_signal2():
    return signal2

def process1():
    rpc.init_rpc(...)
    sleep(10)                              # do some initialization
    some_signal = True
    
    # synchronized
   
def process2():
    rpc.init_rpc(...)
    sleep(5)                              # do some initialization
    while not rpc.rpc_sync("proc1", get_signal1):
        sleep(1)
    # synchronized

def process2():
    rpc.init_rpc(...)
    while (not rpc.rpc_sync("proc1", get_signal1) or
           not rpc.rpc_sync("proc1", get_signal2)):
        sleep(1)
    # synchronized

signals = {}
def lookup(signal_name: str):
   return signals.get(signal_name, None)

torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(0, 1800), backend=None)

torch.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None)
torch.distributed.rpc.rpc_async(to, func, args=None, kwargs=None)
torch.distributed.rpc.remote(to, func, args=None, kwargs=None)

rpc.init_rpc(auto=True)

server = group.get_paired(server_name).to_here()

rpc.init_rpc(auto=True)

rpc.init_rpc("some_name_1")

rpc.init_rpc(auto=True)

# some process may be assigned:
# rpc.get_worker_info().name = "manager:0"
# rpc.get_worker_info().role = "manager"
# rpc.get_worker_info().role_id = 0

if rpc.get_worker_info().role == "manager":
    master_main()
elif rpc.get_worker_info().role == "trainer":
    trainer_main()

self.group.pair(server_name, self)
self.group.register(server_name + "/_push_service", self._push_service)
self.group.register(server_name + "/_pull_service", self._pull_service)