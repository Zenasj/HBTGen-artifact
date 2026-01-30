if CUDNN_HOME is not None:
                extra_ldflags.append(os.path.join(CUDNN_HOME, 'lib/x64'))

if CUDNN_HOME is not None:
                extra_ldflags.append(f"/LIBPATH:{os.path.join(CUDNN_HOME, 'lib/x64')}")

import megatron.fused_kernels
megatron.fused_kernels.load()