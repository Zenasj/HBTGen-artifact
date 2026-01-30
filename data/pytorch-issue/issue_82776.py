loss.backward(retain_graph=True) # This is the backward that recomputes the buffers
loss.backward() # this is the next backward