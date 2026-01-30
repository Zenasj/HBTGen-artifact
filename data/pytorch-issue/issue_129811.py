node = op(input_node, weight_node)

node.meta[NUMERIC_DEBUG_HANDLE_KEY] = {input_node: id1, weight_node: id2, "output": id3}

node.meta[NUMERIC_DEBUG_HANDLE_KEY] = id1