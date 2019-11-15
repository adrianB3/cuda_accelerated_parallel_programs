
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ // the function is a kernel now
void add(int n, float* x, float* y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main()
{
	int N = 1 << 20;
	float* x, * y;

	// allocate unified memory -- accesible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// init the arrays on the host
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	// run kernel on 1M elements on the GPU
	add <<<numBlocks, blockSize>>> (N, x, y);
	
	// wait for gpu to finish
	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}
