#ifndef CUDA_METHODS_H
#define CUDA_METHODS_H

#include <cuda_runtime.h>

float dotProductCPU(float* a, float* b, int n);
__global__ void dotProductGPU(float* a, float* b, float* result, int n);

void allocateGPUMemory(float** d_a, float** d_b, float** d_result, int N);
void copyToGPU(float* d_a, float* d_b, float* h_a, float* h_b, int N, float* d_result, float* h_result);
void copyToCPU(float* h_result, float* d_result);
void freeGPUMemory(float* d_a, float* d_b, float* d_result);
float executeGPU(float* d_a, float* d_b, float* d_result, int N);
void runTests();

#endif