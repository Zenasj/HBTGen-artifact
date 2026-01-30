if "unbind" in proxy.node.name:
     proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val[0])