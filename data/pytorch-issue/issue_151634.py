nodes.sort(
   key=lambda x: V.graph.sizevars.size_hint(
       x.divisor, fallback=config.unbacked_symint_fallback
   )
)