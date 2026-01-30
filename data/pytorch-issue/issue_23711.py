import torch

class MyScriptModule(torch.jit.ScriptModule):
        def __init__(self):
            super(MyScriptModule, self).__init__()
        @torch.jit.script_method
        def forward(self, input_fs1:List[str], input_fs2:str, input_fs3:bool)->Tuple[List[str],str]:
            if input_fs3:
                res_list = [""] * len(input_fs1)
                for i in range(len(input_fs1)):
                    res_list[i] = (input_fs1[i] + input_fs2)
                return (res_list, "input is True")
            else:
                res_list = [""] * len(input_fs1)
                for i in range(len(input_fs1)):
                    res_list[i] = (input_fs1[i] + input_fs2)
                return (res_list, "input is False")
m = MyScriptModule()
script_src = m.code

print(script_src)

def forward(input_fs1,
    input_fs2: str,
    input_fs3: bool) -> Tuple[List[str], str]:
  if input_fs3:
    res_list = torch.mul([""], torch.len(input_fs1))
    for i in range(torch.len(input_fs1)):
      _1 = torch.add(torch.select(input_fs1, i), input_fs2)
      _2 = torch._set_item(res_list, i, _1)
    _0 = (res_list, "input is True")
  else:
    res_list0 = torch.mul([""], torch.len(input_fs1))
    for i0 in range(torch.len(input_fs1)):
      _3 = torch.add(torch.select(input_fs1, i0), input_fs2)
      _4 = torch._set_item(res_list0, i0, _3)
    _0 = (res_list0, "input is False")
  return _0