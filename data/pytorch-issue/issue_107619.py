import torch

class FP16Planner(DefaultSavePlanner):
     def create_local_plan(self):
        plan = super().create_local_plan()
        for p in plan:
            if p.tensor_data is not None:
                p.tensor_data.properties.dtype = torch.float16

class FP16Planner(DefaultSavePlanner):
     def create_local_plan(self):
        plan = super().create_local_plan()
        for p in plan:
            if p.tensor_data is not None:
                p.tensor_data.properties.dtype = torch.float16
        return plan