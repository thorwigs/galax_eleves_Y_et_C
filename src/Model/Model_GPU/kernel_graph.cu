#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "kernel_graph.cuh"
#include <iostream>
#include <cmath>
#define DIFF_T (0.1f)
#define EPS (1.0f)


void create_graph(float4* positionsGPU, float3* velocitiesGPU, int n_particles, cudaGraphExec_t graphExec, cudaStream_t stream, cudaGraph_t graph)
{
	std::cout << "Creating graph..." << std::endl;
	int nthreads = 1024;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	cudaStreamCreate(&stream);

	cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

	compute_acc<<<nblocks, nthreads, nthreads*sizeof(float4)>>>(positionsGPU, velocitiesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);

	cudaStreamEndCapture(stream, &graph);
	cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
	
	std::cout << "Graph created." << std::endl;
}

void update_position_cu_graph(cudaGraphExec_t graphExec, cudaStream_t stream)
{
	std::cout << "Launching graph..." << std::endl;
	cudaGraphLaunch(graphExec, stream);
	std::cout << "Graph launched." << std::endl;
	cudaStreamSynchronize(stream);
}


#endif // GALAX_MODEL_GPU
