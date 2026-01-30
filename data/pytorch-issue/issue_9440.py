# pip install psutil first
import psutil, os
p = psutil.Process( os.getpid() )
old_dlls = set([dll.path for dll in p.memory_maps()])
import torch
new_dlls = set([dll.path for dll in p.memory_maps()])
diff_dlls = new_dlls.difference(old_dlls)
print('\n'.join(diff_dlls))