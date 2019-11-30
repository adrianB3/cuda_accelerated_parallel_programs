#pragma once
#include <string>
#include "tensor.hpp"
class NNLayer
{
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Tensor& forward(Tensor& A) = 0;
	virtual Tensor& backprop(Tensor& dz, float learning_rate) = 0;

	std::string getName() { return this->name; };
};

