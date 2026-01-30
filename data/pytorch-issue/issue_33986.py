with torch.cuda.profiler.profile():
         with torch.autograd.profiler.emit_nvtx():
            main()

import torch
prof = torch.autograd.profiler.load_nvprof("file.nvprof")