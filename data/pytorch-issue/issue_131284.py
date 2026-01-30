import torch.nn as nn

# Clean ipython session
from ctypes import cdll
cdll.LoadLibrary('libcudart.so.12')  # Fails
cdll.LoadLibrary('libnvJitLink.so.12')  # Fails

# Clean ipython session
import torch
from ctypes import cdll
cdll.LoadLibrary('libcudart.so.12')  # Succeeds
cdll.LoadLibrary('libnvJitLink.so.12')  # Fails

cuda_libs: Dict[str, str] = {                                                                                                                                                                                                       
            'cublas': 'libcublas.so.*[0-9]',                                                                                                                                                                                                
            'cudnn': 'libcudnn.so.*[0-9]',                                                                                                                                                                                                  
            'cuda_nvrtc': 'libnvrtc.so.*[0-9]',                                                                                                                                                                                             
            'cuda_runtime': 'libcudart.so.*[0-9]',                                                                                                                                                                                          
            'cuda_cupti': 'libcupti.so.*[0-9]',                                                                                                                                                                                             
            'cufft': 'libcufft.so.*[0-9]',                                                                                                                                                                                                  
            'curand': 'libcurand.so.*[0-9]',                                                                                                                                                                                                
            'nvjitlink': 'libnvJitLink.so.*[0-9]',                                                                                                                                                                                          
            'cusparse': 'libcusparse.so.*[0-9]',                                                                                                                                                                                            
            'cusolver': 'libcusolver.so.*[0-9]',                                                                                                                                                                                            
            'nccl': 'libnccl.so.*[0-9]',                                                                                                                                                                                                    
            'nvtx': 'libnvToolsExt.so.*[0-9]',                                                                                                                                                                                              
        }