#include "cudaMethods.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cuda_runtime.h>

#define BLOCKSIZE 256

float dotProductCPU(float* a, float* b, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

__global__ void dotProductGPU(float* a, float* b, float* result, int n) {
	extern __shared__ float sdata[];
	//уникальный индекс потока относительно сетки
	int tid = threadIdx.x + blockIdx.x * blockDim.x; //локальный индекс в текущем блоке + индекс блока в сетке * размер блока
																									//глобальное смещение
	int local_tid = threadIdx.x;

	float temp = 0;


	if (tid < n)
		temp += a[tid] * b[tid];

	sdata[local_tid] = temp;
	__syncthreads();

	// Выполняем параллельную редукцию внутри блока
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (local_tid < s) {
			sdata[local_tid] += sdata[local_tid + s];
		}
		__syncthreads();
	}

	if (local_tid == 0) {
		atomicAdd(result, sdata[0]);
	}
}

//выделение памяти на гпу
void allocateGPUMemory(float** d_a, float** d_b, float** d_result, int N) {
	cudaMalloc((void**)d_a, N * sizeof(float));
	cudaMalloc((void**)d_b, N * sizeof(float));
	cudaMalloc((void**)d_result, sizeof(float));
}

void copyToGPU(float* d_a, float* d_b, float* h_a, float* h_b, int N, float* d_result, float* h_result) {
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);
}

void copyToCPU(float* h_result, float* d_result) {
	cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
}

void freeGPUMemory(float* d_a, float* d_b, float* d_result) {
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
}

float executeGPU(float* d_a, float* d_b, float* d_result, int N) {
	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	cudaEventRecord(startGPU);

	int blockSize = BLOCKSIZE;
	int gridSize = (N + blockSize - 1) / blockSize;

	dotProductGPU <<< gridSize, blockSize >>> (d_a, d_b, d_result, N);
	cudaEventRecord(stopGPU);

	cudaEventSynchronize(stopGPU);
	float gpuTime = 0.0f;
	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

	cudaEventDestroy(startGPU);
	cudaEventDestroy(stopGPU);

	return gpuTime;
}



CPUResult executeСPU(float* h_a, float* h_b, int N) {
	auto startCPU = std::chrono::high_resolution_clock::now();
	float cpuResult = dotProductCPU(h_a, h_b, N);
	auto endCPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;

	CPUResult result;
	result.result = cpuResult;
	result.duration = durationCPU.count();

	return result;
}
