#pragma once

#include "nnlayer.h"

class SigmoidActivation : public NNLayer
{
private:
	Tensor2D A;
	Tensor2D Z;
	Tensor2D dZ;

public:
	SigmoidActivation(std::string name);

	~SigmoidActivation();

	Tensor2D& forward(Tensor2D& Z);
	Tensor2D& backprop(Tensor2D& dA, float learning_rate = 0.01);
};

