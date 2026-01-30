def _get_vc_env(vc_arch: str) -> dict[str, str]:
    try:
        from setuptools import distutils  # type: ignore[import]

        return distutils._msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]
    except AttributeError:
        from setuptools._distutils import _msvccompiler  #type: ignore[import]

        return _msvccompiler._get_vc_env(vc_arch)  # type: ignore[no-any-return]