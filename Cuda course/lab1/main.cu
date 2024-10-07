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

void vectInit(float*& h_a, float*& h_b, int N) {
	for (int i = 0; i < N; i++) {
		h_a[i] = static_cast<float>(rand()) / RAND_MAX;
		h_b[i] = static_cast<float>(rand()) / RAND_MAX;
	}
}

void ExecuteGPU(int N, float*& h_a, float*& h_b, float& h_result, std::vector<float>& gpuTimes) {
	float* d_a, * d_b, * d_result;

	allocateGPUMemory(&d_a, &d_b, &d_result, N);
	copyToGPU(d_a, d_b, h_a, h_b, N, d_result, &h_result);

	float gpuTime = executeGPU(d_a, d_b, d_result, N);

	gpuTimes.push_back(gpuTime);
	copyToCPU(&h_result, d_result);

	freeGPUMemory(d_a, d_b, d_result);
}

void CalcDifference(float& cpuResult, float& h_result, std::vector<float>& absDifferences, std::vector<float>& resultsCPU, std::vector<float>& resultsGPU) {
	float absDifference = fabs(cpuResult - h_result);
	absDifferences.push_back(absDifference);
	resultsCPU.push_back(cpuResult);
	resultsGPU.push_back(h_result);
}

void executeAll(int N, std::vector<int>& sizes, std::vector<float>& resultsCPU, std::vector<float>& resultsGPU, std::vector<float>& cpuTimes, std::vector<float>& gpuTimes, std::vector<float>& absDifferences) {
	sizes.push_back(N);

	float* h_a = new float[N];
	float* h_b = new float[N];
	vectInit(h_a, h_b, N);

	CPUResult cpuResult = executeСPU(h_a, h_b, N);
	cpuTimes.push_back(cpuResult.duration);

	float h_result = 0.0f;
	ExecuteGPU(N, h_a, h_b, h_result, gpuTimes);
	CalcDifference(cpuResult.result, h_result, absDifferences, resultsCPU, resultsGPU);

	delete[] h_a;
	delete[] h_b;
}

void printResults(std::vector<int>& sizes, std::vector<float>& resultsCPU, std::vector<float>& resultsGPU, std::vector<float>& cpuTimes, std::vector<float>& gpuTimes, std::vector<float>& absDifferences) {
	std::cout << std::left << std::setw(10) << "Size"
		<< std::setw(15) << "CPU Time (ms)"
		<< std::setw(15) << "GPU Time (ms)"
		<< std::setw(15) << "SpeedUp"
		<< std::setw(20) << "Result difference"
		<< std::setw(15) << "CPU result"
		<< std::setw(15) << "GPU result" << std::endl;

	for (size_t i = 0; i < sizes.size(); ++i) {
		std::cout << std::fixed << std::setprecision(5)
			<< std::setw(10) << sizes[i]
			<< std::setw(15) << cpuTimes[i]
			<< std::setw(15) << gpuTimes[i]
			<< std::setw(15) << cpuTimes[i] / gpuTimes[i]
			<< std::setw(20) << absDifferences[i]
			<< std::setw(15) << resultsCPU[i]
			<< std::setw(15) << resultsGPU[i]
			<< std::endl;
	}
}

void Test(int N) {
	std::vector<int> sizes;
	std::vector<float> resultsCPU, resultsGPU;
	std::vector<float> cpuTimes, gpuTimes, absDifferences;
	std::cout << "Test Started\n\n";
	executeAll(N, sizes, resultsCPU, resultsGPU, cpuTimes, gpuTimes, absDifferences);
	printResults(sizes, resultsCPU, resultsGPU, cpuTimes, gpuTimes, absDifferences);

	if(absDifferences[0] < 0.01)
		std::cout << "\nTest Passed\n";
	else
		std::cout << "\nTest Failed\n";

	std::cout << "\nTest Ended\n\n";
}

int main() {
	Test(1000);

	std::vector<int> sizes;
	std::vector<float> resultsCPU, resultsGPU;
	std::vector<float> cpuTimes, gpuTimes, absDifferences;

	for (int N = STEP; N <= MAX_N; N += STEP) {
		executeAll(N, sizes, resultsCPU, resultsGPU, cpuTimes, gpuTimes, absDifferences);
	}

	printResults(sizes, resultsCPU, resultsGPU, cpuTimes, gpuTimes, absDifferences);


	return 0;
}