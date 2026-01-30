import sys
import torch
import torch._inductor
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only

aten = torch.ops.aten

from torch._inductor.pattern_matcher import (
   CallFunction,
   KeywordArg,
   MultiOutputPattern,
)
    
match_pattern = MultiOutputPattern([CallFunction(aten.sub.Tensor, KeywordArg("a"), KeywordArg("b"))])

def test_replacement1(a, b):
    return a + b

def test_replacement2(a, b):
    return a * b     
                     
def test_pattern(a, b):
    return a - b     
    
def binary_nop(a, b):
    return 1
    
if sys.argv[1] == '1':
    print("using add")
    rep = test_replacement1
else:
    print("using mul")
    rep = test_replacement2
    
a = torch.empty((2,2))
b = a
my_patterns = PatternMatcherPass()
register_replacement(binary_nop,  # just to get arg names?                                                                        
                     rep,
                     [a, b],
                     fwd_only,
                     [my_patterns],
                     search_fn_pattern=match_pattern)
    
def custom_pass(graph):
    my_patterns.apply(graph)

def test_backend(graph, example_inputs):
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)
                     
@torch.compile(backend=test_backend)
def test(a, b):
    return a - b
    
a = torch.ones((2, 2)) * 3
b = torch.ones((2, 2)) * 2
c = test(a, b)
print(c)