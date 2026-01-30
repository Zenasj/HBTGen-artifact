state_dict = real_model.state_dict()
state_dict.update(dict(real_model.named_buffers()))  # Save non-persistent buffers along with the rest of the model state_dict