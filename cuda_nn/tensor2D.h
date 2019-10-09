#pragma once
#include "shape.h"
#include<memory>
class Tensor2D
{
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Shape shape;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	Tensor2D(size_t x_dim = 1, size_t y_dim = 1);
	Tensor2D(Shape shape);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	float& operator[](const int index);
	const float& operator[](const int index) const;
};

