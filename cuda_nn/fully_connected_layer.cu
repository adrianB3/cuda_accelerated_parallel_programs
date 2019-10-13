#include "fully_connected_layer.h"
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fullyConnectedLayerForward(float* W, float* A, float* Z, float* b, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int Z_x_dim = A_x_dim;
	int Z_y_dim = W_y_dim;

	float Z_value = 0;

	if (row < Z_y_dim && col < Z_x_dim)
	{
		for (int i = 0; i < W_x_dim; i++)
		{
			Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
		}
		Z[row * Z_x_dim + col] = Z_value + b[row];
	}
}

__global__ void fullyConnectedLayerBackprop(float* W, float* dZ, float* dA, int W_x_dim, int W_y_dim, int dZ_x_dim, int dZ_y_dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int dA_x_dim = dZ_x_dim;
	int dA_y_dim = W_x_dim;

	float dA_value = 0.0f;
	
	if (row < dA_y_dim && col < dA_x_dim)
	{
		for (int i = 0; i < W_y_dim; i++)
		{
			dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		}
		dA[row * dA_x_dim + col] = dA_value;
	}
}

void FullyConnectedLayer::initializeBiasWithZeros()
{
	for (int x = 0; x < b.shape.x; x++)
	{
		b[x] = 0;
	}

	b.copyHostToDevice();
}

void FullyConnectedLayer::initializeWeightsRandomly()
{
	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);

	for (int x = 0; x < W.shape.x; x++)
	{
		for (int y = 0; y < W.shape.y; y++)
		{
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

FullyConnectedLayer::FullyConnectedLayer(std::string name, Shape W_shape)
	:W(W_shape), b(W_shape.y, 1)
{
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros();
	initializeWeightsRandomly();
}

FullyConnectedLayer::~FullyConnectedLayer()
{
}

int FullyConnectedLayer::getXDim() const
{
	return W.shape.x;
}

int FullyConnectedLayer::getYDim() const
{
	return W.shape.y;
}

Tensor2D FullyConnectedLayer::getWeightsMatrix() const
{
	return W;
}

Tensor2D FullyConnectedLayer::getBiasVector() const
{
	return b;
}
