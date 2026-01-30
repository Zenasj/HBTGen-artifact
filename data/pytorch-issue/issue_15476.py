# The script will prompt the user to specify ``CUDA_TOOLKIT_ROOT_DIR`` if
# the prefix cannot be determined by the location of nvcc in the system
# path and ``REQUIRED`` is specified to :command:`find_package`.  To use
# a different installed version of the toolkit set the environment variable
# ``CUDA_BIN_PATH`` before running cmake (e.g.
# ``CUDA_BIN_PATH=/usr/local/cuda1.0`` instead of the default
# ``/usr/local/cuda``) or set ``CUDA_TOOLKIT_ROOT_DIR`` after configuring.  If
# you change the value of ``CUDA_TOOLKIT_ROOT_DIR``, various components that
# depend on the path will be relocated.
#
# It might be necessary to set ``CUDA_TOOLKIT_ROOT_DIR`` manually on certain
# platforms, or to use a CUDA runtime not installed in the default
# location.  In newer versions of the toolkit the CUDA library is
# included with the graphics driver -- be sure that the driver version
# matches what is needed by the CUDA runtime version.

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.