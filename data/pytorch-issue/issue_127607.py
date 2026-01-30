for layer in self.layers.values():
    h = layer(h, self.freqs_cis)

for node in call_module_nodes:
        node.args = tuple(filter(lambda n: n.name not in inputs_to_state, node.args))