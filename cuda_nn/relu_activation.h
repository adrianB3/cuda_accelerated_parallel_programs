#pragma once
#include "nnlayer.h"
class ReLUActivation : public NNLayer
{
private:
	Tensor2D A;
	Tensor2D Z;
	Tensor2D dZ;

public:
	ReLUActivation(std::string name);

	~ReLUActivation();

	Tensor2D& forward(Tensor2D& Z);
	Tensor2D& backprop(Tensor2D& dA, float learning_rate = 0.01);
};

