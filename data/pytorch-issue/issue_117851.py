@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]
    name: str

    def __call__(self, *args):
        CleanupManager.count -= 1
        del self.scope[self.name]

    @staticmethod
    def create(scope, name, val):
        assert name not in scope
        CleanupManager.count += 1
        scope[name] = val
        return CleanupHook(scope, name)

def create_load_python_module(self, mod, push_null) -> Instruction:
        """
        Generate a LOAD_GLOBAL instruction to fetch a given python module.
        """
        global_scope = self.tx.output.global_scope
        name = re.sub(r"^.*[.]", "", mod.__name__)
        if global_scope.get(name, None) is mod:
            return self.create_load_global(name, push_null, add=True)
        mangled_name = f"___module_{name}_{id(mod)}"
        if mangled_name not in global_scope:
            self.tx.output.install_global(mangled_name, mod)
        return self.create_load_global(mangled_name, push_null, add=True)