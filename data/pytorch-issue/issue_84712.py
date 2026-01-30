from distutils.version import LooseVersion

if not hasattr(tensorboard, "__version__") or LooseVersion(
    tensorboard.__version__
) < LooseVersion("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del LooseVersion

import packaging

if not hasattr(tensorboard, "__version__") or packaging.version.parse(
        tensorboard.__version__
) < packaging.version.Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del packaging