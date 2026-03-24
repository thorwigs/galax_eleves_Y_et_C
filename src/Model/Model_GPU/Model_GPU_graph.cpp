#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU_graph.hpp"
#include "kernel_graph.cuh"


inline bool cuda_malloc(void ** devPtr, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to allocate buffer" << std::endl;
		return false;
	}
	return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst, src, count, kind);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to copy buffer" << std::endl;
		return false;
	}
	return true;
}

void update_position_gpu_graph(cudaGraphExec_t graphExec, cudaStream_t stream)
{
	update_position_cu_graph(graphExec, stream);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}


Model_GPU_graph
::Model_GPU_graph(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  positionsf3    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;

	for (int i = 0; i < n_particles; i++)
	{
		positionsf3[i].x     = initstate.positionsx [i];
		positionsf3[i].y     = initstate.positionsy [i];
		positionsf3[i].z     = initstate.positionsz [i];
		positionsf3[i].w     = initstate.masses     [i];
		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
	}

	cuda_malloc((void**)&positionsGPU,         n_particles * sizeof(float4));
	cuda_malloc((void**)&velocitiesGPU,        n_particles * sizeof(float3));

	cuda_memcpy(positionsGPU,      positionsf3.data(),           n_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cuda_memcpy(velocitiesGPU,     velocitiesf3.data(),          n_particles * sizeof(float3), cudaMemcpyHostToDevice);

	create_graph(positionsGPU, velocitiesGPU, n_particles, graphExec, stream, graph);
}

Model_GPU_graph
::~Model_GPU_graph()
{
	cudaFree((void**)&positionsGPU);
	cudaFree((void**)&velocitiesGPU);
	cudaFree((void**)&accelerationsGPU);
}

void Model_GPU_graph
::step()
{
	update_position_gpu_graph(graphExec, stream);

	cuda_memcpy(positionsf3.data(), positionsGPU, n_particles * sizeof(float4), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = positionsf3[i].x;
		particles.y[i] = positionsf3[i].y;
		particles.z[i] = positionsf3[i].z;
	}
}

#endif // GALAX_MODEL_GPU
