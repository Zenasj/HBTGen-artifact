import torch

centernesses = [centerness.sigmoid() for centerness in centernesses]
centernesses = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]
centernesses = [torch.chunk(centerness, num_imgs, dim = 0) for centerness in centernesses]
#print('centernesses (in forward_nms):')
#print(centernesses)
centernesses = [ torch.cat(list(map(lambda x: x[i], centernesses)), dim = 0)
                         for i in range(num_imgs)]

centernesses = [ torch.cat(list(map(lambda x: x[i], centernesses)), dim = 0)
                         for i in range(num_imgs)]

frame #0: c10::Error::Error(c10::SourceLocation, std::string const&) + 0x33 (0x2b584b18f193 in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0x17f66 (0x2b584af4af66 in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libc10_cuda.so)
frame #2: <unknown function> + 0x19cbd (0x2b584af4ccbd in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10::TensorImpl::release_resources() + 0x4d (0x2b584b17f63d in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libc10.so)
frame #4: <unknown function> + 0x67bac2 (0x2b57ffe69ac2 in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libtorch_python.so)
frame #5: <unknown function> + 0x67bb66 (0x2b57ffe69b66 in /home/jhli/project/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0x19dfce (0x561487cb5fce in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #7: <unknown function> + 0x113a6b (0x561487c2ba6b in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #8: <unknown function> + 0x113bc7 (0x561487c2bbc7 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #9: <unknown function> + 0x103948 (0x561487c1b948 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #10: <unknown function> + 0x114267 (0x561487c2c267 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #11: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #12: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #13: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #14: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #15: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #16: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #17: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #18: <unknown function> + 0x11427d (0x561487c2c27d in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #19: PyDict_SetItem + 0x502 (0x561487c77602 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #20: PyDict_SetItemString + 0x4f (0x561487c780cf in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #21: PyImport_Cleanup + 0x9e (0x561487cb791e in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #22: Py_FinalizeEx + 0x67 (0x561487d2d367 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #23: <unknown function> + 0x227d93 (0x561487d3fd93 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #24: _Py_UnixMain + 0x3c (0x561487d400bc in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)
frame #25: __libc_start_main + 0xf5 (0x2b57ea8eb505 in /lib64/libc.so.6)
frame #26: <unknown function> + 0x1d0990 (0x561487ce8990 in /home/jhli/project/.conda/envs/open-mmlab2/bin/python)