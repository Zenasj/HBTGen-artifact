tmp = tensor.new_empty([0]).set_(tensor.storage())
tmp.record_stream(stream)