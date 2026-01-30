import torch
import pickle
class CustomUnpickler(pickle.Unpickler):
    def load(self):
        raise
class CustomPickle:
    Unpickler = CustomUnpickler
torch.save({'nothing', None}, 'test_pickle.pt')
torch.load('test_pickle.pt', pickle_module=CustomPickle)