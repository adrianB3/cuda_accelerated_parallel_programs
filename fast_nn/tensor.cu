#include "tensor.hpp"
#include <cuda_runtime.h>

void Tensor::allocateCudaMemory()
{
	if (!device_allocated)
	{
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		data_device = std::shared_ptr<float>(device_memory, [&](float* ptr) {cudaFree(ptr); });
		device_allocated = true;
	}
}

void Tensor::allocateHostMemory()
{
	if (!host_allocated)
	{
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y], [&](float* ptr) { delete[] ptr; });
		host_allocated = true;
	}
}

Tensor::Tensor(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{}

Tensor::Tensor(Shape shape) :
	Tensor(shape.x, shape.y)
{}

void Tensor::allocateMemory()
{
	allocateCudaMemory();
	allocateHostMemory();
}

void Tensor::allocateMemoryIfNotAllocated(Shape shape)
{
	if (!device_allocated && !host_allocated)
	{
		this->shape = shape;
		allocateMemory();
	}
}

float& Tensor::operator[](const int index)
{
	return data_host.get()[index];
}

const float& Tensor::operator[](const int index) const
{
	return data_host.get()[index];
}