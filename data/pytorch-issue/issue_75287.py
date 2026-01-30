def load_state_dict_post_hook(module, incompatible_keys):
    deprecated_keys = {"foo", "bar"}
    incompatible_keys.unexpected_keys[:] = [
        key for key in incompatible_keys.unexpected_keys
        if key.split(".")[-1] not in deprecated_keys
    ]

def load_state_dict_post_hook(module, incompatible_keys, prefix):
    deprecated_keys = ["foo", "bar"]
    for key in deprecated_keys:
        if prefix + key in incompatible_keys.unexpected_keys:
            incompatible_keys.unexpected_keys.remove(prefix + key)

def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    deprecated_keys = ["foo", "bar"]
    for key in deprecated_keys:
        if prefix + key in state_dict:
            del state_dict[prefix + key]
    super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        deprecated_keys = ["weights", "_float_tensor"]
        for key in deprecated_keys:
            if prefix + key in state_dict:
                del state_dict[prefix + key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)