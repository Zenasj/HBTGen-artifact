import torch

def is_package_import_successful() -> bool:
    try:
        import pandas
    except ImportError:
        package_import = None
    return package_import is not None

def check_import():
    a = torch.add(2, 3)
    if is_package_import_successful():
        print ("Pandas import is successful")
    else:
        print ("Pandas import not successful")


check_import()
print ("Compilation started.")
compile_import = torch.compile(check_import)
print ("Compilation done")
compile_import()