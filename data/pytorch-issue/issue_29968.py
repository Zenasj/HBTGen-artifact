#include "c10/util/numa.h"

C10_DEFINE_bool(caffe2_cpu_numa_enabled, false, "Use NUMA whenever possible.");

#if defined(__linux__) && !defined(C10_DISABLE_NUMA) && !defined(C10_MOBILE)
#include <numa2.h>
#include <numaif.h>