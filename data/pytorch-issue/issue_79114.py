model = FSDP(model, auto_wrap=...)
model.register_comm_hook(state, hook) # shared across all FSDP submodules

def register_comm_hook(state, hook):
    self.hook = hook
    self.state = state
    # share hook and state with all submodules