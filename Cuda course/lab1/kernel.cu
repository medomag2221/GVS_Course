#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
#include <cassert>

#define MAX_N 10000000  // Максимальный размер векторов
#define STEP 1000000    // Шаг увеличения размера векторов

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

// Функция для тестирования корректности
void testDotProduct() {
    const int N = 1000; // Небольшой размер для тестирования
    float* h_a = new float[N];
    float* h_b = new float[N];

    // Инициализация векторов
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    // Ожидаемый результат
    float expectedResult = dotProductCPU(h_a, h_b, N);

    // Создание и инициализация векторов на GPU
    float* d_a, * d_b, * d_result;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    float h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Запуск ядра
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    dotProductGPU << <gridSize, blockSize >> > (d_a, d_b, d_result, N);

    // Копирование результата обратно на CPU
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    float absDifference = fabs(expectedResult - h_result);
    if (absDifference < 0.00001f) {
        std::cout << "Test passed" << std::endl;
    }
    // Освобождение памяти
    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    std::cout << "Test failed" << std::endl;

}

int main() {
    // Запускаем тест
    testDotProduct();

    // Основная часть программы
    std::vector<int> sizes;
    std::vector<float> resultsCPU, resultsGPU;
    std::vector<float> cpuTimes, gpuTimes, absDifferences;

    for (int N = STEP; N <= MAX_N; N += STEP) {
        sizes.push_back(N);

        // Инициализация векторов на CPU
        float* h_a = new float[N];
        float* h_b = new float[N];
        for (int i = 0; i < N; i++) {
            h_a[i] = static_cast<float>(rand()) / RAND_MAX;
            h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Замер времени выполнения на CPU
        auto startCPU = std::chrono::high_resolution_clock::now();
        float cpuResult = dotProductCPU(h_a, h_b, N);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
        cpuTimes.push_back(durationCPU.count());

        // Создаем и инициализируем векторы на GPU
        float* d_a, * d_b, * d_result;
        cudaMalloc((void**)&d_a, N * sizeof(float));
        cudaMalloc((void**)&d_b, N * sizeof(float));
        cudaMalloc((void**)&d_result, sizeof(float));

        cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

        float h_result = 0.0f;
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

        // Замер времени выполнения на GPU
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
        gpuTimes.push_back(gpuTime);


        // Копируем результат обратно на CPU
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        // Вычисление модуля разности между результатами на CPU и GPU
        float absDifference = fabs(cpuResult - h_result);
        absDifferences.push_back(absDifference);
        resultsCPU.push_back(cpuResult);
        resultsGPU.push_back(h_result);
        // Освобождаем память
        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaEventDestroy(startGPU);
        cudaEventDestroy(stopGPU);
    }

    // Выводим результаты
    std::cout << "Size, CPU Time (ms), GPU Time (ms), Abs Difference (result), CPU result, GPU result" << std::endl;
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::fixed << std::setprecision(10) << sizes[i] << ", " << cpuTimes[i] << ", " << gpuTimes[i] << ", " << absDifferences[i] << ", " << resultsCPU[i] << ", " << resultsGPU[i] << std::endl;
    }

    return 0;
}