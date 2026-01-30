try:
    state_dict = model.module.state_dict()
except AttributeError:
    state_dict = model.state_dict()