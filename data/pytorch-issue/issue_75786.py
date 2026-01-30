py
sys.path.insert(0, path)

# do something

sys.path.remove(path)

py
import sys


path = "/tmp"

# PR
try:
    sys.path.insert(0, path)
    try:
        # Any exception raised while performing the actual functionality
        raise Exception
    finally:
        sys.path.remove(path)
except Exception:
    assert path not in sys.path

# main
try:
    sys.path.insert(0, path)

    # Any exception raised while performing the actual functionality
    raise Exception

    sys.path.remove(path)
except Exception:
    assert path in sys.path