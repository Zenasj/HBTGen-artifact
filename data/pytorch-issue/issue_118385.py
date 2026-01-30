def free_symbols(self) -> Set[sympy.Symbol]:
        return set(self.var_to_val.keys()) - set(self.replacements.keys())