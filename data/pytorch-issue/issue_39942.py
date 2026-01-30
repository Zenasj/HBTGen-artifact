# before backward / directly after loss calculation: tensor([0, 1, 2])
# after backward: tensor([-5340268357666996224, -5340268354615407047, 0, 1, 2, 3051589177])

self._storage = data_sample

self._storage = None

...

# load sample, forward, store