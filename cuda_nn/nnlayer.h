#pragma once
#include <string>
#include "tensor2D.h"
class NNLayer
{
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Tensor2D& forward(Tensor2D& A) = 0;
	virtual Tensor2D& backprop(Tensor2D& dz, float learning_rate) = 0;

	std::string getName() { return this->name; };
};

