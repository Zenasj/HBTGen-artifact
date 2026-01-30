import torch.nn as nn

if __name__ == '__main__':
    from lightning.fabric import Fabric

    fabric = Fabric(
            accelerator="gpu",
            devices=2,
            strategy="ddp",
            precision="16-mixed"
        )

    main()

import os
import sys
import torch
from nvitop import Device
import subprocess

def print_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("Error running nvidia-smi:", result.stderr)
        else:
            print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi command not found. Please make sure NVIDIA drivers are installed and nvidia-smi is in your PATH.")

def print_process_info():
    devices = Device.cuda.all()
    for device in devices:
        processes = device.processes() 
        sorted_pids = sorted(processes.keys())
        print(f'Processes ({len(processes)}): {sorted_pids}')
        for pid in sorted_pids:
            print(f'\t- {processes[pid]}')

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDNN version: {torch.backends.cudnn.version()}")
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print_nvidia_smi_output()
print_process_info()

# smoke test
print(f'torch.empty(2, device="cuda"): {torch.empty(2, device="cuda")}')

import os
import sys
import torch
import subprocess
import re

def get_mig_uuids():
    result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Command 'nvidia-smi -L' failed with exit code {result.returncode}")
    
    output = result.stdout
    print(output)

    mig_uuid_pattern = re.compile(r'MIG-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
    
    mig_uuids = mig_uuid_pattern.findall(output)
    
    return mig_uuids

def set_cuda_visible_devices(mig_uuids):
    mig_uuids_str = ','.join(mig_uuids)
    os.environ['CUDA_VISIBLE_DEVICES'] = mig_uuids_str
    print(f"CUDA_VISIBLE_DEVICES set to: {mig_uuids_str}")

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDNN version: {torch.backends.cudnn.version()}")
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

mig_uuids = get_mig_uuids()
if mig_uuids:
    set_cuda_visible_devices(mig_uuids)
else:
    print("No MIG devices found.")

# smoke tests
print(f"torch.randn(1).cuda(): {torch.randn(1).cuda()}")
print(f'torch.empty(2, device="cuda"): {torch.empty(2, device="cuda")}')