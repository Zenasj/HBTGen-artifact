if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            key = str(node.meta["original_aten"]._overloadpacket) # THIS ONE