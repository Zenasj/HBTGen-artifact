import torch

state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
                        else torch.tensor(0.)  # <-- no device here
                    )

state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
                        else torch.tensor(0., device=p.device)  # <-- no device here
                    )