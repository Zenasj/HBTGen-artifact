import torch

from setuptools import setup, find_packages                                                
from torch.utils.cpp_extension import BuildExtension, CUDAExtension                        
                                                                                            
setup(name='stnls',                                                                        
           package_dir={"": "lib"},                                                             
           packages=find_packages("."),                                                         
          package_data={'': ['*.so']},                                                         
          include_package_data=True,                                                           
          ext_modules=[                                                                        
              CUDAExtension('my_name_is_arch', [                                                    
               'lib/csrc/my_file.cpp',
               'lib/csrc/my_file_kernel.cu',])
               ],
          cmdclass={'build_ext': BuildExtension},
         )