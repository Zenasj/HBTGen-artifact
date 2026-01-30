import torch
import torch.nn as nn
import sys
import gc

ref_counter = "Some weird string"
base_rc = sys.getrefcount(ref_counter)
def print_rc():
    print(sys.getrefcount(ref_counter)-base_rc)

class SomeNet(nn.Module):

    def __init__(self, smth=None):
        super(SomeNet, self).__init__()
        self.smth = smth
        #generates some memory usage (to see the problem):
        self.memoryUser = nn.Sequential(
            nn.Linear(10000, 256),
            nn.Linear(256, 200)
        )
        self.compare = ["nice object", "totally nice object"]
        self.wired = ["nice object", "totally nice object"]

        self.someLambda = lambda x: ( self.wired )

    def forward(self, x):
        return x

print("Our refcounter to track when our module dies")
print_rc()
o = SomeNet(ref_counter)
print("Our module owns one ref")
print_rc()
del o
print("It did not go away after del?")
print_rc()
gc.collect()
print("What if we run the gc to collect ref cycles?")
print_rc()
print("Victory !")