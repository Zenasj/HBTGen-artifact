from typing import casting, Optional
rocm_home=cast(Optional[str], os.path.dirname(os.path.dirname(
    os.path.realpath(os.path.abspath(shutil.which("hipcc"))))))