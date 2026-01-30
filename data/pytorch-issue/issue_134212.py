from typing import cast
from torch.optim import AdamW 
from torch.optim.optimizer import _use_grad_for_differentiable
from torch.optim.adamw import adamw
import torch 
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

def _group_mixed_tensor_dtensors(params: list[torch.Tensor], grads: list[torch.Tensor], *states: list[torch.Tensor]):
    same_placement_tensors: list[torch.Tensor] = []
    same_placement_grads: list[torch.Tensor] = []
    same_placement_states: list[list[torch.Tensor]] = [[] for _ in states]
    device_mesh_to_tensors: dict[DeviceMesh | None, list[torch.Tensor]] = {}
    device_mesh_to_grads: dict[DeviceMesh | None, list[torch.Tensor]] = {}
    device_mesh_to_states: dict[DeviceMesh | None, list[list[torch.Tensor]]] = {}
    assert len(params) == len(grads)
    cnt = 0
    for param, grad in zip(params, grads):
        if isinstance(param, DTensor):
            assert isinstance(grad, DTensor)
            if param.placements == grad.placements:
                # param placement will never be partial, so we can safety convert
                # them to local if they are DTensor.
                same_placement_tensors.append(param.to_local())
                same_placement_grads.append(grad.to_local())
                for i, state in enumerate(states):
                    state_ten = state[cnt]
                    if isinstance(state_ten, DTensor):
                        same_placement_states[i].append(state_ten.to_local())
                    else:
                        same_placement_states[i].append(state_ten)
            else:
                device_mesh = param.device_mesh
                if device_mesh not in device_mesh_to_tensors:
                    device_mesh_to_tensors[device_mesh] = []
                    device_mesh_to_grads[device_mesh] = []
                    device_mesh_to_states[device_mesh] = [[] for _ in states]
                device_mesh_to_tensors[device_mesh].append(param)
                device_mesh_to_grads[device_mesh].append(grad)
                for i, state in enumerate(states):
                    device_mesh_to_states[device_mesh][i].append(state[cnt])
        else:
            if None not in device_mesh_to_tensors:
                device_mesh_to_tensors[None] = []
                device_mesh_to_grads[None] = []
                device_mesh_to_states[None] = [[] for _ in states]
            device_mesh_to_tensors[None].append(param)
            device_mesh_to_grads[None].append(grad)
            for i, state in enumerate(states):
                device_mesh_to_states[None][i].append(state[cnt])
        cnt += 1
    return ([same_placement_tensors] + list(device_mesh_to_tensors.values()), 
            [same_placement_grads] + list(device_mesh_to_grads.values()),
            [same_placement_states] + list(device_mesh_to_states.values()))

class AdamWMixed(AdamW):
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(tuple[float, float], group["betas"])

            has_complex = self._init_group(...)
            if amsgrad:
                grouped_res = _group_mixed_tensor_dtensors(params_with_grad, grads, exp_avgs, exp_avg_sqs,state_steps, max_exp_avg_sqs)
            else:
                grouped_res = _group_mixed_tensor_dtensors(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps)
            num_groups = len(grouped_res[0])
            for j in range(num_groups):
                params_with_grad = grouped_res[0][j]
                if len(params_with_grad) == 0:
                    continue
                grads = grouped_res[1][j]
                exp_avgs = grouped_res[2][j][0]
                exp_avg_sqs = grouped_res[2][j][1]
                state_steps = grouped_res[2][j][2]
                if amsgrad:
                    max_exp_avg_sqs = grouped_res[2][j][3]
                
                adamw(...)

        return loss