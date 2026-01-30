def register_module(self, name: str, module: Optional['Module']) -> None:
    r"""Alias for :func:`add_module`."""
    self.add_module(name, module)