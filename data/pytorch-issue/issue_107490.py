class RenamePlanner(DefaultSavePlanner):
    def set_up_planner(self, state_dict, is_coordinator):
        # prefix all keys with `foo_``
        super().set_up_planner(self, {"foo_" + k: v for k, v in state_dict.items()}, is_coordinator)

class RenamePlanner(DefaultSavePlanner):
    def set_up_planner(self, state_dict, is_coordinator):
        # prefix all keys with `foo_``
        super().set_up_planner({"foo_" + k: v for k, v in state_dict.items()}, is_coordinator)