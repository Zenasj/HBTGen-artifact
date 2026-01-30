import torch
import torch.nn as nn

def test_ModuleList(self):
        def func2():
            modules = [nn.ReLU(), nn.Linear(5, 5)]
            module_list = nn.ModuleList(modules)
            del module_list[1::2]
            return module_list
        


        @torch.compile()
        def func1():
            modules1 = [nn.ReLU(), nn.Linear(5, 5)]
            module_list1 = nn.ModuleList(modules1)
            del module_list1[1::2]
            return module_list1


        print(func1())
        print(func2())
        # self.assertEqual(func1(), func2())

def test_ModuleList(self):
        def func2():
            modules1 = [nn.ReLU(), nn.Linear(5, 5)]
            module_list1 = nn.ModuleList(modules1)
            for k in range(2)[1::2]:
                del module_list1._modules[str(k)]
            # del module_list1[1::2]
            return module_list1


        @torch.compile(backend="eager")
        def func1():
            modules1 = [nn.ReLU(), nn.Linear(5, 5)]
            module_list1 = nn.ModuleList(modules1)
            for k in range(2)[1::2]:
                del module_list1._modules[str(k)]
            # del module_list1[1::2]
            return module_list1


        print(func1())
        print(func2())

for k in range(2)[1::2]:
                comptime.print(k)