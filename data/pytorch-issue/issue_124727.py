#70 201.0 + sudo -E -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/envs/py_3.8/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LD_LIBRARY_PATH= conda run -n py_3.8 pip install --progress-bar off git+https://github.com/openai/triton@45fff310c891f5a92d55445adf8cc9d29df5841e#subdirectory=python
#70 266.2   Running command git clone --filter=blob:none --quiet https://github.com/openai/triton /tmp/pip-req-build-7cggbwil
#70 266.2   fatal: the remote end hung up unexpectedly
#70 266.2   fatal: early EOF
#70 266.2   fatal: index-pack failed
#70 266.2   error: subprocess-exited-with-error
#70 266.2   
#70 266.2   × git clone --filter=blob:none --quiet https://github.com/openai/triton /tmp/pip-req-build-7cggbwil did not run successfully.
#70 266.2   │ exit code: 128
#70 266.2   ╰─> See above for output.
#70 266.2   
#70 266.2   note: This error originates from a subprocess, and is likely not a problem with pip.
#70 266.2 error: subprocess-exited-with-error
#70 266.2 
#70 266.2 × git clone --filter=blob:none --quiet https://github.com/openai/triton /tmp/pip-req-build-7cggbwil did not run successfully.
#70 266.2 │ exit code: 128
#70 266.2 ╰─> See above for output.
#70 266.2 
#70 266.2 note: This error originates from a subprocess, and is likely not a problem with pip.
#70 266.2 
#70 266.2 ERROR conda.cli.main_run:execute(124): `conda run pip install --progress-bar off git+[https://github.com/openai/triton@45fff310c891f5a92d55445adf8cc9d29df5841e#subdirectory=python`](https://github.com/openai/triton@45fff310c891f5a92d55445adf8cc9d29df5841e#subdirectory=python%60) failed. (See above for error)
#70 266.2 Collecting git+https://github.com/openai/triton@45fff310c891f5a92d55445adf8cc9d29df5841e#subdirectory=python
#70 266.2   Cloning https://github.com/openai/triton (to revision 45fff310c891f5a92d55445adf8cc9d29df5841e) to /tmp/pip-req-build-7cggbwil
#70 266.2 
#70 ERROR: process "/bin/sh -c if [ -n \"${TRITON}\" ]; then bash ./install_triton.sh; fi" did not complete successfully: exit code: 1