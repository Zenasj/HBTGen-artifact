# grep 'lmpi' CMakeCache.txt 
MPI_CXX_LINK_FLAGS:STRING=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi_cxx;-lmpi
MPI_C_LINK_FLAGS:STRING=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi
MPI_CXX_PKG_LDFLAGS:INTERNAL=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi_cxx;-lmpi
MPI_CXX_PKG_STATIC_LDFLAGS:INTERNAL=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi_cxx;-lmpi;-lopen-rte;-lopen-pal;-lhwloc;-ldl;-lutil;-lm
MPI_C_PKG_LDFLAGS:INTERNAL=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi
MPI_C_PKG_STATIC_LDFLAGS:INTERNAL=-L/usr/lib/x86_64-linux-gnu/openmpi/lib;-L/usr//lib;-lmpi;-lopen-rte;-lopen-pal;-lhwloc;-ldl;-lutil;-lm