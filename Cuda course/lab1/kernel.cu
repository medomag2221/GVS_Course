#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
#include <cassert>
#include "cudaMethods.h"

#define MAX_N 10000000
#define STEP 1000000

int main() {


	//тесты
	runTests();



	std::vector<int> sizes;
	std::vector<float> resultsCPU, resultsGPU;
	std::vector<float> cpuTimes, gpuTimes, absDifferences;

	for (int N = STEP; N <= MAX_N; N += STEP) {
		sizes.push_back(N);

		//инициализация векторов
		float* h_a = new float[N]; 
		float* h_b = new float[N];
		for (int i = 0; i < N; i++) {
			h_a[i] = static_cast<float>(rand()) / RAND_MAX;
			h_b[i] = static_cast<float>(rand()) / RAND_MAX;
		}
		
		//замер времени
		auto startCPU = std::chrono::high_resolution_clock::now();
		float cpuResult = dotProductCPU(h_a, h_b, N);
		auto endCPU = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
		cpuTimes.push_back(durationCPU.count());

		// выделяем память под векторы на гпу
		float* d_a, * d_b, * d_result;
		allocateGPUMemory(&d_a, &d_b, &d_result, N);

		float h_result = 0.0f;
		copyToGPU(d_a, d_b, h_a, h_b, N, d_result, &h_result);

		// мерим время гпу
		float gpuTime = executeGPU(d_a, d_b, d_result, N);
		gpuTimes.push_back(gpuTime);

		// забираем результаты с гпу
		copyToCPU(&h_result, d_result);

		// считаем разницу
		float absDifference = fabs(cpuResult - h_result);
		absDifferences.push_back(absDifference);
		resultsCPU.push_back(cpuResult);
		resultsGPU.push_back(h_result);

		// освобождаем память
		delete[] h_a;
		delete[] h_b;
		freeGPUMemory(d_a, d_b, d_result);
	}


	//результаты
	std::cout << std::left << std::setw(10) << "Size"
		<< std::setw(20) << "CPU Time (ms)"
		<< std::setw(20) << "GPU Time (ms)"
		<< std::setw(25) << "Abs Difference (result)"
		<< std::setw(20) << "CPU result"
		<< std::setw(20) << "GPU result" << std::endl;

	for (size_t i = 0; i < sizes.size(); ++i) {
		std::cout << std::fixed << std::setprecision(5)
			<< std::setw(10) << sizes[i]
			<< std::setw(20) << cpuTimes[i]
			<< std::setw(20) << gpuTimes[i]
			<< std::setw(25) << absDifferences[i]
			<< std::setw(20) << resultsCPU[i]
			<< std::setw(20) << resultsGPU[i]
			<< std::endl;
	}

	return 0;
}