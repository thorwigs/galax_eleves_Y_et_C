#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_graph_HPP_
#define MODEL_GPU_graph_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel_graph.cuh"

class Model_GPU_graph : public Model
{
private:

	std::vector<float4> positionsf3    ;
	std::vector<float3> velocitiesf3   ;
	std::vector<float3> accelerationsf3;

	cudaGraph_t graph;
	cudaGraphExec_t graphExec;
	cudaStream_t stream;

	float4* positionsGPU;
	float3* velocitiesGPU;
	float3* accelerationsGPU;

public:
	Model_GPU_graph(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU_graph();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
