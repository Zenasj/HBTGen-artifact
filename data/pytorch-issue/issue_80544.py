import torch

# torch.Tensors cannot be used as a key in a dictionary
# because they define a custom __eq__ function which when used
# to resolve hash collisions will throw when comparing tensors:
# "RuntimeError: bool value of Tensor with more than one value is ambiguous."
# To avoid that, we use an object which will hold a Tensor and use
# its id for both hashing and equality.
# In order to use this as a weak key reference, we cannot
# simply use weakref.WeakKeyDictionary because the newly constructed
# WeakTensorRefKey only use would be a dictionary so it would have no strong
# references.
# To get around this issue, we can use it as a normal key, and then set
# `weakref.finalize` to delete the key when its contained tensor dies.

# [expired-storages]
# NB: even though the tensor has died,
# the deallocation of its storage can take longer,
# even when the storage has no other uses/views.
# In this case, the StorageWeakRef object will be kept alive
# longer than it needs to be, however the storage itself
# will be deallocated. We retain the possibly dead storages
# and periodically check if any of them are expired and
# can be freed.