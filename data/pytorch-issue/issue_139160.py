import torch

py
def forward(self, x):
    return x.sum() + self.attr_list[-1].sum()
# No break, no error.

py
def forward(self, x):
    self.attr_list.append(torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[0].sum()
# No break, no error.

py
def forward(self, x):
    self.attr_list.append(torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[1].sum()
# `[ERROR]: fullgraph compilation failed: 'FakeRootModule' object has no attribute 'self___attr_list_0'

py
def forward(self, x):
    self.attr_list.insert(0, torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[0].sum()
# No break, no error

py
def forward(self, x):
    self.attr_list.insert(0, torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[-1].sum()
# No break, no error

py
def forward(self, x):
    self.attr_list.insert(-1, torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[0].sum()
# No break, no error

py
def forward(self, x):
    self.attr_list.insert(-1, torch.ones(3, 2).to('cuda:0'))
    return x.sum() + self.attr_list[-1].sum()
# [ERROR]: fullgraph compilation failed: 'FakeRootModule' object has no attribute 'self___attr_list_0'