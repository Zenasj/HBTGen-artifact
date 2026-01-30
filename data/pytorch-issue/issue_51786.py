import torch

OpInfo('pow',
           op=torch.pow,
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_pow,
           test_inplace_grad=False,
           assert_autodiffed=True,
           supports_tensor_out=True,
           decorators=(precisionOverride({torch.float16: 1e-1, torch.bfloat16: 1e-1, 
                                          torch.float32: 1e-5, torch.float64: 1e-5}),)),