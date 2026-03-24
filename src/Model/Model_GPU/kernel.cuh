#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, int n_particles);
__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, int n_particles);

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, int n_particles);
#endif

#endif // GALAX_MODEL_GPU
