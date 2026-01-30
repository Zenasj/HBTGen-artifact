import torch.nn as nn
import torch.jit as jit


class SomeScriptModule(jit.ScriptModule):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(16, 16)
        self.conv2d = nn.Conv2d(3, 8, 3)
        self.conv3d = nn.Conv3d(3, 8, 3)
        self.gru = nn.GRU(16, 16)
        self.lstm = nn.LSTM(16, 16)

        for m in self.modules():
            print(m, type(m))

            if isinstance(m, nn.Linear):
                print(f'm is Linear')
                continue

            if isinstance(m, nn.Conv2d):
                print(f'm is Conv2d')
                continue

            if isinstance(m, nn.Conv3d):
                print(f'm is Conv3d')
                continue

            if isinstance(m, nn.GRU):
                print(f'm is GRU')
                continue

            if isinstance(m, nn.LSTM):
                print(f'm is LSTM')
                continue

            print('??????')


SomeScriptModule()

class JITModule(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 1, 1)])

        self.reset_parameters()

    def reset_parameters(self):
        for name, m in self.named_modules():
            if 'convs' in name:
                print('JITModule:', name, m, isinstance(m, nn.Conv2d))

    @torch.jit.script_method
    def forward(self, x):
        for m in self.convs:
            x = m(x)
        return x


class ParentModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = JITModule()


for name, m in ParentModule().named_modules():
    if 'convs' in name:
        print('ParentModule:', name, m, isinstance(m, nn.Conv2d))