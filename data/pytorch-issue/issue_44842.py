import torch.nn as nn

import torch

class Foo(torch.nn.Module):
    @property
    def prop(self):
        return self._prop


    @prop.setter
    def prop(self, val):
        some_long_continued_string = f"""\
dedent here lol
        """
        self._prop = some_long_continued_string


f = Foo()

torch.jit.script(f)