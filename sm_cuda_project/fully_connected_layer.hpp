#pragma once
#include "nnlayer.h"
class FullyConnectedLayer : public NNLayer
{
private:
	const float weights_init_threshold = 0.01;

	Tensor2D W, b, Z, A, dA;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeAndStoreBackpropError(Tensor2D& dZ);
	void computeAndStoreLayerOutput(Tensor2D& A);
	void updateWeights(Tensor2D& dZ, float learning_rate);
	void updateBias(Tensor2D& dZ, float learning_rate);

public:
	FullyConnectedLayer(std::string name, Shape W_shape);
	~FullyConnectedLayer();

	Tensor2D& forward(Tensor2D& A);
	Tensor2D& backprop(Tensor2D& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Tensor2D getWeightsMatrix() const;
	Tensor2D getBiasVector() const;
};

