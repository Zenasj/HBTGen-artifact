# this makes self.min_val have the device of state_dict['min_val']
self.min_val = state_dict['min_val']

# this preserves the device of `self.min_val`
self.min_val.copy_(state_dict['min_val'])