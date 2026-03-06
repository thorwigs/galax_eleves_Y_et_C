#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_MIX_HPP_
#define MODEL_GPU_MIX_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"

class Model_GPU_MIX : public Model
{
private:

	std::vector<float3> positionsf3    ;
	std::vector<float3> velocitiesf3   ;
	std::vector<float3> accelerationsf3;

	float3* positionsGPU;
	float3* velocitiesGPU;
	float3* accelerationsGPU;
	float*  massesGPU;

public:
	Model_GPU_MIX(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU_MIX();

	virtual void step();
};
#endif // MODEL_GPU_MIX_HPP_

#endif // GALAX_MODEL_GPU
