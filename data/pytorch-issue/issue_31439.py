import cppyy
cppyy.load_library('glog')
cppyy.include('glog/logging.h')
cppyy.gbl.google.InstallFailureSignalHandler()
import scipy
import torch