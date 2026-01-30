import ctypes

dll_files = ["vcruntime140.dll", "msvcp140.dll", "vcruntime140_1.dll"]

try:
    for dll in dll_files:
        ctypes.CDLL(dll)
    print(f"DLLs loaded successfully")
except OSError as e:
    print(f"Error loading {dll}: {e}")