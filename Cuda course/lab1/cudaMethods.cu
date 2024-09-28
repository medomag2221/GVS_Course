#include "cudaMethods.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cuda_runtime.h>


// Реализация скалярного произведения на CPU
float dotProductCPU(float* a, float* b, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

// Ядро для скалярного произведения на GPU
__global__ void dotProductGPU(float* a, float* b, float* result, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float temp = 0;
	while (tid < n) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	atomicAdd(result, temp);
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
	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	dotProductGPU << <gridSize, blockSize >> > (d_a, d_b, d_result, N);
	cudaEventRecord(stopGPU);

	cudaEventSynchronize(stopGPU);
	float gpuTime = 0.0f;
	cudaEventElapsedTime(&gpuTime, startGPU, stopGPU);

	cudaEventDestroy(startGPU);
	cudaEventDestroy(stopGPU);

	return gpuTime;
}

void runTests() {
	const int smallSize = 1000;   // Маленький тест
	const int largeSize = 100000; // Большой тест

	// Маленький тест для проверки корректности
	float* h_a = new float[smallSize];
	float* h_b = new float[smallSize];
	for (int i = 0; i < smallSize; i++) {
		h_a[i] = static_cast<float>(i);
		h_b[i] = static_cast<float>(i + 1);
	}

	// Вычисление на CPU
	auto startCPU = std::chrono::high_resolution_clock::now();
	float cpuResult = dotProductCPU(h_a, h_b, smallSize);
	auto endCPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
	std::cout << "CPU Time for small size: " << durationCPU.count() << " ms" << std::endl;

	// Выделение памяти на GPU
	float* d_a, * d_b, * d_result;
	allocateGPUMemory(&d_a, &d_b, &d_result, smallSize);

	float h_result = 0.0f;

	// Копирование данных на GPU
	copyToGPU(d_a, d_b, h_a, h_b, smallSize, d_result, &h_result);

	// Измерение времени выполнения на GPU
	float gpuTime = executeGPU(d_a, d_b, d_result, smallSize);
	std::cout << "GPU Time for small size: " << gpuTime << " ms" << std::endl << std::endl;

	// Копирование результата обратно на CPU
	copyToCPU(&h_result, d_result);

	// Вывод для отладки
	std::cout << "CPU result: " << cpuResult << std::endl;
	std::cout << "GPU result: " << h_result << std::endl << std::endl;

	// Проверка корректности результатов
	float absDifference = fabs(cpuResult - h_result);

	assert(absDifference < 100.0f);

	std::cout << "Test passed for small size, absolute difference: " << absDifference << std::endl << std::endl;

	freeGPUMemory(d_a, d_b, d_result);

	h_a = new float[largeSize];
	h_b = new float[largeSize];
	for (int i = 0; i < largeSize; i++) {
		h_a[i] = static_cast<float>(rand()) / RAND_MAX;
		h_b[i] = static_cast<float>(rand()) / RAND_MAX;
	}

	startCPU = std::chrono::high_resolution_clock::now();
	cpuResult = dotProductCPU(h_a, h_b, largeSize);
	endCPU = std::chrono::high_resolution_clock::now();
	durationCPU = endCPU - startCPU;
	std::cout << "CPU Time for large size: " << durationCPU.count() << " ms" << std::endl;

	allocateGPUMemory(&d_a, &d_b, &d_result, largeSize);
	h_result = 0.0f;

	copyToGPU(d_a, d_b, h_a, h_b, largeSize, d_result, &h_result);

	gpuTime = executeGPU(d_a, d_b, d_result, largeSize);
	std::cout << "GPU Time for large size: " << gpuTime << " ms" << std::endl << std::endl;

	copyToCPU(&h_result, d_result);

	// Проверка корректности результатов
	absDifference = fabs(cpuResult - h_result);
	std::cout << "CPU result: " << cpuResult << std::endl;
	std::cout << "GPU result: " << h_result << std::endl << std::endl;

	assert(absDifference < 100.0f);  // Допустимая разница для большого теста

	std::cout << "Test passed for large size, absolute difference: " << absDifference << std::endl << std::endl;

	// Освобождаем память
	delete[] h_a;
	delete[] h_b;
	freeGPUMemory(d_a, d_b, d_result);
}