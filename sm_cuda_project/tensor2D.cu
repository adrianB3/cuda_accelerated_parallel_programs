#include "tensor2D.hpp"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

void Tensor2D::allocateCudaMemory()
{
	if (!device_allocated)
	{
		float* device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		data_device = std::shared_ptr<float>(device_memory, [&](float* ptr) {cudaFree(ptr); });
		device_allocated = true;
	}
}

void Tensor2D::allocateHostMemory()
{
	if (!host_allocated)
	{
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y], [&](float* ptr) { delete[] ptr; });
		host_allocated = true;
	}
}

Tensor2D::Tensor2D(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{}

Tensor2D::Tensor2D(Shape shape) :
	Tensor2D(shape.x, shape.y)
{}

void Tensor2D::allocateMemory()
{
	allocateCudaMemory();
	allocateHostMemory();
}

void Tensor2D::allocateMemoryIfNotAllocated(Shape shape)
{
	if (!device_allocated && !host_allocated)
	{
		this->shape = shape;
		allocateMemory();
	}
}

float& Tensor2D::operator[](const int index)
{
	return data_host.get()[index];
}

const float& Tensor2D::operator[](const int index) const
{
	return data_host.get()[index];
}
