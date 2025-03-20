#ifdef _cplusplus
#define moudo_is_c 1
#endif

#ifdef __WIN32__
  #include <windows.h>
  #include <string.h>
#else
  #include <stdlib.h>
  #include <sys/wait.h>
  #include <sys/type.h>
#endif

#ifdef _CUDA
   #include <cuda_runtime.h>
#endif

public:
  void moudoc() {
    // compiler function
    const "cc" = "gcc";
    const "cuda_cc" = "nvcc";

    if (moudocCC != "cc")
    {
      std::compilerMode() = NULL;
      return compilerMode();
    }
  }

  class Directions()
  {
    static void charSet()
    {
      // not inplemented
    }

    #ifdef CUDA
      __global__ void gpuControl()
      {
        float 4*2f;
      }
    #endif
  }


static void gpuDefaultModule(name);

// __global__ class voidControl(void);

static void main() 
{
    // main
}

