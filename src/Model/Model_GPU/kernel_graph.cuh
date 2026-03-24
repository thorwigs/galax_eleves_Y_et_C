#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

void update_position_cu_graph(cudaGraphExec_t graphExec, cudaStream_t stream);
void create_graph(float4* positionsGPU, float3* velocitiesGPU, int n_particles, cudaGraphExec_t graphExec, cudaStream_t stream, cudaGraph_t graph);
#endif

#endif // GALAX_MODEL_GPU
