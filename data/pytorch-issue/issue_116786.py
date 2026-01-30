def access(self, parent: Any) -> Any:
        ...

def key_get(obj: Any, kp: KeyPath) -> Any:
    for k in kp:
        obj = k.access(obj)
    return obj