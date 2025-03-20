#ifndef MOUDOSTDLIB_H
#define MOUDOSTDLIB_H

#ifdef _CUDA_
  #include <cuda_runtime.h>
#endif

#ifdef __WIN32__
  #include <windows.h>
#else
  #include <stdlib.h>
  #include <string.h>
  #include <sys/type.h>
#endif

// implement it later


#endif // MOUDOSTDLIB_H

