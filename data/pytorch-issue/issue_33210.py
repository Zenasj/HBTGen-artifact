@_require_initialized
def typeof(rref):
    return rpc_sync(rref.owner(), _typeof, args = (rref,))

def _typeof(rref):
    return type(rref.local_value())