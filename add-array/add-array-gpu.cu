/****************************************************************************80

    Array element addition using CUDA on GPUs

    Note: changes from the C++ / CPU-only file marked intentionally w/ "CUDA"
          supposed to be *annoyingly* commented for (self-)educational purposes
          "host" is assumed to be the CPU running the C program.
          "device" is assumed to be the GPU running the kernel.
    
    MIT License

    Copyright (c) 2017 Marcelo Novaes

******************************************************************************/

#include <chrono>
#include <iostream>
#include <math.h>

#include <cuda_runtime.h>

/****************************************************************************80

    PRINT_TIME calculates how many seconds since the start of the program

******************************************************************************/

using namespace std::chrono;

const system_clock::time_point start = system_clock::now();

float print_time()
{
    // elapsed time
    system_clock::time_point end = system_clock::now();
 
    duration<double> elapsed_seconds = end - start;
    std::time_t end_time = system_clock::to_time_t(end);
 
    float elapsed = elapsed_seconds.count();

    // print
    printf("@ %06.3fs > ", elapsed);
    return elapsed;
}

/****************************************************************************80

    ADD_ARRAYS adds elements of two arrays on GPU

******************************************************************************/

// CUDA: kernel defined by __global__ declaration specifier, accessed by host & device
// see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
__global__ void add_arrays(int n, float *x, float *y, float *z)
{
    // loop over the array index adding elements *in parallel*
    // CUDA: threads have an ID, reside within a block, reside within a grid
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < n)
    {
        z[i] = x[i] + y[i];
    }
}

/****************************************************************************80

    MAIN is the main program for array element addition on GPU example

******************************************************************************/

int main(void)
{
    print_time();
    // set up array dimensions and byte size
    printf("Setting up array dimension and size...\n");
    int N = pow(10, 8); // 100M elements per array
    size_t sizeN = N * sizeof(float);

    print_time();
    // allocate memory for arrays
    // CUDA: this allocation is happening in the host, thus renamed as `h?`
    printf("Allocating memory for arrays...\n");
    float *hx = (float *)malloc(sizeN);
    float *hy = (float *)malloc(sizeN);
    float *hz = (float *)malloc(sizeN);

    // CUDA: this allocation is happening in the device, thus renamed as `d?`
    // CUDA: allocate Unified Memory to be bridged between the  CPU and GPU
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-explicit-allocation
    float *dx = NULL;
    cudaMalloc(&dx, sizeN);
    float *dy = NULL;
    cudaMalloc(&dy, sizeN);
    float *dz = NULL;
    cudaMalloc(&dz, sizeN);

    print_time();
    // initialize x, y, z arrays
    // CUDA: initializing in the host
    printf("Initializing arrays...\n");
    for (int i = 0; i < N; i++)
    {
        hx[i] = 2.0f;
        hy[i] = 3.0f;
        hz[i] = 0.0f;
        //printf("i = %d, hx = %0.6f, hy = %0.6f, hz = %0.6f\n", i, hx[i], hy[i], hz[i]);
    }

    print_time();
    // CUDA: copies the arrays from host to device
    // CUDA: cudaMemcpy(from, to, size, direction);
    printf("Copying arrays from host to device...\n");
    cudaMemcpy(dx, hx, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz, sizeN, cudaMemcpyHostToDevice);

    print_time();
    // add elements (on GPU)
    // CUDA: run the kernel call, <<<...>>> notation defines # of CUDA blocks/threads
    // e.g.:  <<<16, 256>>> means 16 blocks of 256 threads
    // this alone doesn't parallelize it, must make changes in the kernel function
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
    // see: http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
    int nThreads = 256; // CUDA: Threads per block
    int nBlocks = N / nThreads; // CUDA: Number of blocks on total grid
    printf("Adding %d elements from both arrays...\n", N);
    printf("Kernel using %d threads for each of %d blocks.\n", nThreads, nBlocks);
    add_arrays<<<nBlocks, nThreads>>>(N, dx, dy, dz); // CUDA: kernel call<<<blocks, threads>>>(args);
  
    // CUDA: wait the GPU to finish commands in all streams of all host threads
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#explicit-synchronization
    cudaDeviceSynchronize();
    
    print_time();
    // CUDA: copies the arrays from device back to host
    // CUDA: cudaMemcpy(from, to, size, direction);
    printf("Copying arrays from device back to host...\n");
    cudaMemcpy(hz, dz, sizeN, cudaMemcpyDeviceToHost);

    print_time();
    // sum errors to check for problems
    printf("Aggregating deltas over all elements to check for errors...\n");
    float sumOfErrors = 0.0f;
    for (int i = 0; i < N; i++)
    {
        sumOfErrors += fabs(hx[i] + hy[i] - hz[i]);
    }
    printf("Sum of errors: %.6f\n", sumOfErrors);

    // print first and last z array contents
    printf("hz[0] = %.6f\n", hz[0]);
    printf("hz[N-1] = %.6f\n", hz[N-1]);

    print_time();
    // free mem when done
    printf("Freeing memory...\n");
    delete [] hx;
    delete [] hy;
    delete [] hz;
    // CUDA: device memory can be allocated as linear memory or as CUDA arrays
    // linear memory usually allocated using cudaMalloc(). freed using cudaFree()
    // see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    // C'est fini
    print_time();
    printf("Done.\n");

    return 0;
}
