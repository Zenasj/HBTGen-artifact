import torch

for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype()) # LINE 97
                )
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype()) # Here
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state['step']):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype(), device=p.device) #Similar code
                    if not torch.is_tensor(p_state["eta"]):
                        p_state["eta"] = torch.tensor(p_state["eta"], dtype=_get_scalar_dtype(), device=p.device)
                    if not torch.is_tensor(p_state["mu"]):
                        p_state["mu"] = torch.tensor(p_state["mu"], dtype=_get_scalar_dtype(), device=p.device)

p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype(), device=p.device)

s["step"] = torch.tensor(float(s["step"]), dtype=_get_scalar_dtype(), device=p.device)