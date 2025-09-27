#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    } 
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    //host
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    //intialize host
    for (int i=0; i<N; i++) {
        h_a[i] = i+10;
        h_b[i] = i*40;
    }

    // device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //kernel config
    int blockSize = 256;  //number of threads inside block
    int gridSize = (N + blockSize - 1) / blockSize; //number of blocks

    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    //copy result back from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //verify
    for (int i=0; i<25; i++) {
        printf("a[%d] + b[%d] = %.2f + %.2f = %.2f\n", 
            i, i, h_a[i], h_b[i], h_c[i]);
    }
    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
