#17 128.1 + sudo -E -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env PATH=/opt/conda/bin:/opt/conda/envs/py_3.12/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 conda run -n py_3.12 pip install --progress-bar off -r /opt/conda/requirements-ci.txt
#17 159.6   error: subprocess-exited-with-error
#17 159.6   
#17 159.6   × Preparing metadata (pyproject.toml) did not run successfully.
#17 159.6   │ exit code: 1
#17 159.6   ╰─> [16215 lines of output]
#17 159.6       + meson setup /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9 /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/.mesonpy-nfj2tdja -Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md --native-file=/tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/.mesonpy-nfj2tdja/meson-python-native-file.ini
#17 159.6       The Meson build system
#17 159.6       Version: 1.5.1
#17 159.6       Source dir: /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9
#17 159.6       Build dir: /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/.mesonpy-nfj2tdja
#17 159.6       Build type: native build
#17 159.6       Project name: scikit-image
#17 159.6       Project version: 0.20.0
#17 159.6       C compiler for the host machine: cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
#17 159.6       C linker for the host machine: cc ld.bfd 2.34
#17 159.6       C++ compiler for the host machine: c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
#17 159.6       C++ linker for the host machine: c++ ld.bfd 2.34
#17 159.6       Host machine cpu family: x86_64
#17 159.6       Host machine cpu: x86_64
#17 159.6       Compiler for C supports arguments -Wno-unused-function: YES
#17 159.6       Library m found: YES
#17 159.6       Checking if "-Wl,--version-script" : links: YES
#17 159.6       Program cython found: YES (/tmp/pip-build-env-hg_eh9yk/overlay/bin/cython)
#17 159.6       Program pythran found: YES (/tmp/pip-build-env-hg_eh9yk/overlay/bin/pythran)
#17 159.6       Program cp found: YES (/usr/bin/cp)
#17 159.6       Program python found: YES (/opt/conda/envs/py_3.12/bin/python)
#17 159.6       Found pkg-config: YES (/usr/bin/pkg-config) 0.29.1
#17 159.6       Run-time dependency python found: YES 3.12
#17 159.6       Program _build_utils/cythoner.py found: YES (/tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/skimage/_build_utils/cythoner.py)
#17 159.6       Compiler for C++ supports arguments -Wno-cpp: YES
#17 159.6       Build targets in project: 58
#17 159.6       
#17 159.6       scikit-image 0.20.0
#17 159.6       
#17 159.6         User defined options
#17 159.6           Native files: /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/.mesonpy-nfj2tdja/meson-python-native-file.ini
#17 159.6           buildtype   : release
#17 159.6           b_ndebug    : if-release
#17 159.6           b_vscrt     : md
#17 159.6       
#17 159.6       Found ninja-1.11.1.git.kitware.jobserver-1 at /tmp/pip-build-env-hg_eh9yk/normal/bin/ninja
#17 159.6       + /tmp/pip-build-env-hg_eh9yk/normal/bin/ninja
#17 159.6       [1/168] Generating skimage/morphology/_skeletonize_3d_cy with a custom command
#17 159.6       [2/168] Compiling C object skimage/restoration/_unwrap_2d.cpython-312-x86_64-linux-gnu.so.p/unwrap_2d_ljmu.c.o
#17 159.6       [3/168] Generating skimage/feature/_hessian_det_appx_pythran with a custom command
#17 159.6       FAILED: skimage/feature/_hessian_det_appx.cpp
#17 159.6       /tmp/pip-build-env-hg_eh9yk/overlay/bin/pythran -E ../skimage/feature/_hessian_det_appx_pythran.py -o skimage/feature/_hessian_det_appx.cpp
#17 159.6       Traceback (most recent call last):
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/bin/pythran", line 8, in <module>
#17 159.6           sys.exit(run())
#17 159.6                    ^^^^^
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/run.py", line 190, in run
#17 159.6           pythran.compile_pythranfile(args.input_file,
#17 159.6           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/__init__.py", line 127, in __getattr__
#17 159.6           import pythran.toolchain
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/toolchain.py", line 11, in <module>
#17 159.6           from pythran.dist import PythranExtension, PythranBuildExt
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/dist.py", line 142, in <module>
#17 159.6           class PythranBuildExt(PythranBuildExtMixIn, LegacyBuildExt, metaclass=PythranBuildExtMeta):
#17 159.6       TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
#17 159.6       [4/168] Generating skimage/feature/_brief_pythran with a custom command
#17 159.6       FAILED: skimage/feature/brief_cy.cpp
#17 159.6       /tmp/pip-build-env-hg_eh9yk/overlay/bin/pythran -E ../skimage/feature/brief_pythran.py -o skimage/feature/brief_cy.cpp
#17 159.6       Traceback (most recent call last):
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/bin/pythran", line 8, in <module>
#17 159.6           sys.exit(run())
#17 159.6                    ^^^^^
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/run.py", line 190, in run
#17 159.6           pythran.compile_pythranfile(args.input_file,
#17 159.6           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/__init__.py", line 127, in __getattr__
#17 159.6           import pythran.toolchain
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/toolchain.py", line 11, in <module>
#17 159.6           from pythran.dist import PythranExtension, PythranBuildExt
#17 159.6         File "/tmp/pip-build-env-hg_eh9yk/overlay/lib/python3.12/site-packages/pythran/dist.py", line 142, in <module>
#17 159.6           class PythranBuildExt(PythranBuildExtMixIn, LegacyBuildExt, metaclass=PythranBuildExtMeta):
#17 159.6       TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
#17 159.6       [5/168] Generating 'skimage/io/_plugins/_histograms.cpython-312-x86_64-linux-gnu.so.p/_histograms.c'
#17 159.6       [6/168] Generating 'skimage/_shared/fast_exp.cpython-312-x86_64-linux-gnu.so.p/fast_exp.c'
#17 159.6       [7/168] Generating 'skimage/io/_plugins/_colormixer.cpython-312-x86_64-linux-gnu.so.p/_colormixer.c'
#17 159.6       performance hint: /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/skimage/io/_plugins/_colormixer.pyx:213:5: Exception check on 'rgb_2_hsv' will always require the GIL to be acquired.
#17 159.6       Possible solutions:
#17 159.6           1. Declare 'rgb_2_hsv' as 'noexcept' if you control the definition and you're sure you don't want the function to raise exceptions.
#17 159.6           2. Use an 'int' return type on 'rgb_2_hsv' to allow an error code to be returned.
#17 159.6       performance hint: /tmp/pip-install-wpy3l459/scikit-image_2ba4226c0cca42e684d7d04da673fef9/skimage/io/_plugins/_colormixer.pyx:276:5: Exception check on 'hsv_2_rgb' will always require the GIL to be acquired.
#17 159.6       Possible solutions: