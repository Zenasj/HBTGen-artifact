import torch

class RenamePlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, metadata, is_coordinator):
        self.original_state_dict = state_dict
        super().set_up_planner(self, {"foo_" + k: v for k, v in state_dict.items()}, is_coordinator)
    def load_bytes(self, read_item, value):
        # Remove the "foo_" prefix
        self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value)