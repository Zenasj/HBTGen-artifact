# _wait_tensor(self) returns self directly, so this is the same as syncing + running `buf3 = buf2`
buf3 = _wait_tensor(buf2); del buf2

# All future operations (views, inputs to other ops, etc) should use buf3, not buf2