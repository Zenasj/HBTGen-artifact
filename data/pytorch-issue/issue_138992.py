import os
import sys
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"

def test_case():
    cwd = os.getcwd()
    path1 = os.path.join(cwd, "haha1.txt")
    path2 = Path(os.path.join(cwd, "haha2.txt"))

    try:
        path2.rename(path1)
    except FileExistsError as e_file_exist:
        if _IS_WINDOWS:
            # on Windows file exist is expected: https://docs.python.org/3/library/pathlib.html#pathlib.Path.rename
            shutil.copy2(path2, path1)
            os.remove(path2)
        else:
            raise e_file_exist
    except BaseException as e:
        raise e

    print("run here.")

if __name__ == "__main__":
    test_case()

shutil.copy2(src=tmp_path, dst=path)
os.remove(tmp_path)