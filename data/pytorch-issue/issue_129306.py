import torch
import sys
sys.path.append("/path/to/cloned_repo/pytorch-3dunet/")
from pytorch3dunet.unet3d.model import get_model


model_config = {'name': 'UNet3D', 'in_channels': 1, 
                'out_channels': 1, 'layer_order': 'gcr', 
                'f_maps': 32, 'num_groups': 8, 'final_sigmoid': True}


def main():
    # Create model
    model = get_model(model_config)
    model = torch.compile(model)
    print("#### Model Loaded")
    image = torch.rand(1,1,80,170,170)

    from torch.profiler import ExecutionTraceObserver
    et = ExecutionTraceObserver()
    et.register_callback(f"unet3d_traces.json")
    et.start()

    print("#### Model Running")
    output = model(image)

    et.stop()
    et.unregister_callback()

    print("Model Ran Successfully")

if __name__ == '__main__':
    main()

[tasklist]
### Tasks