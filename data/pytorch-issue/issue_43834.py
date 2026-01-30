from ctypes import cdll

def load(x):
    print("Loading", x)
    try:
        cdll.LoadLibrary(x)
    except Exception as e:
        print("\tFailed", e)
        pass
    else:
        print("\tSucc")

load("/opt/app-root/lib/python3.6/site-packages/torch/_C.cpython-36m-x86_64-linux-gnu.so")
#load("/opt/app-root/lib/python3.6/site-packages/cv2/../opencv_python.libs/libz-d8a329de.so.1.2.7")
load("/opt/app-root/lib/python3.6/site-packages/cv2/../opencv_python.libs/libcrypto-354cbd1a.so.1.1")
load("libgcc_s.so.1")