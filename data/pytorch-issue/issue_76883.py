import argparse
import torch
import torch.utils.jit.log_extract as log_extract

ir = ["""graph(%pad_sequence_embeddings_7.1 : Half(10, 260, 192, strides=[49920, 192, 1], requires_grad=0, device=cuda:0),
      %pad_sequence_embeddings_3.1 : Half(10, 1, 192, strides=[192, 192, 1], requires_grad=0, device=cuda:0)):
  %2 : int = prim::Constant[value=1]()
  %6 : Half(10, 192, strides=[192, 1], requires_grad=0, device=cuda:0) = aten::squeeze(%pad_sequence_embeddings_3.1, %2)
  %7 : Half(10, 1, 192, strides=[192, 192, 1], requires_grad=0, device=cuda:0) = aten::unsqueeze(%6, %2)
  %mul_1.1 : Half(10, 260, 192, strides=[49920, 192, 1], requires_grad=0, device=cuda:0) = aten::mul(%pad_sequence_embeddings_7.1, %7)
  return (%mul_1.1, %7)
""", """graph(%weight.132 : Half(1024, strides=[1], requires_grad=0, device=cuda:0),
      %bias.78 : Half(1024, strides=[1], requires_grad=0, device=cuda:0),
      %bmm_9.1 : Half(10, 1024, 1, strides=[1024, 1, 1], requires_grad=0, device=cuda:0)):
  %3 : bool = prim::Constant[value=1]()
  %4 : float = prim::Constant[value=1.0000000000000001e-05]()
  %5 : int[] = prim::Constant[value=[1024]]()
  %6 : int = prim::Constant[value=-1]()
  %11 : Half(10, 1024, strides=[1024, 1], requires_grad=0, device=cuda:0) = aten::squeeze(%bmm_9.1, %6)
  %main_module_over_arch_1_norm_0.1 : Half(10, 1024, strides=[1024, 1], requires_grad=0, device=cuda:0) = aten::layer_norm(%11, %5, %weight.132, %bias.78, %4, %3)
  %main_module_over_arch_1_norm_1.1 : Half(10, 1024, strides=[1024, 1], requires_grad=0, device=cuda:0) = aten::sigmoid(%main_module_over_arch_1_norm_0.1)
  %mul_21.1 : Half(10, 1024, strides=[1024, 1], requires_grad=0, device=cuda:0) = aten::mul(%11, %main_module_over_arch_1_norm_1.1)
  return (%mul_21.1)
"""]

parser = argparse.ArgumentParser(description='Type the index of the ir graph to run, or -1 to run all')
parser.add_argument('id', type=int)
args = parser.parse_args()

def run_on(ir):
    _, inputs = log_extract.load_graph_and_inputs(ir)
    log_extract.run_nvfuser(ir, inputs)

if args.id == -1:
    for i, s in enumerate(ir):
        print('graph', i)
        run_on(s)
else:
    run_on(ir[args.id])