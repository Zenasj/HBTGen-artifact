def ensure_size_computed(self, sym: sympy.Symbol):
    if isinstance(sym, sympy.Symbol) and symbol_is_type(sym, SymT.PRECOMPUTED_SIZE):
        if sym in self.computed_sizes:
            return
        self.computed_sizes.add(sym)
        expr = V.graph.sizevars.inv_precomputed_replacements[sym]
        self.writeline(
            f"{self.declare}{sym} = {self.expr_printer(expr)}{self.ending}"
        )