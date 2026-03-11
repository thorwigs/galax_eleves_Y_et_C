#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#include <cmath>
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n_particles) {
		return;
	}

	float3 acc = {0.0f, 0.0f, 0.0f};

	for (int j = 0; j < n_particles; j++)
	{
		const float diffx = positionsGPU[j].x - positionsGPU[i].x;
		const float diffy = positionsGPU[j].y - positionsGPU[i].y;
		const float diffz = positionsGPU[j].z - positionsGPU[i].z;

		float dij = fmaf(diffx, diffx, fmaf(diffy, diffy, fmaf(diffz, diffz, 0.0f)));

		if (dij < 1.0)
		{
			dij = 10.0;
		}
		else
		{
			dij = rsqrtf(dij);
			dij = 10.0 * (dij * dij * dij);
		}

		acc.x += diffx * dij * massesGPU[j];
		acc.y += diffy * dij * massesGPU[j];
		acc.z += diffz * dij * massesGPU[j];
	}

	velocitiesGPU[i].x += acc.x * 2.0f;
	velocitiesGPU[i].y += acc.y * 2.0f;
	velocitiesGPU[i].z += acc.z * 2.0f;

}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= n_particles) {
		return;
	}

	// positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	// positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	// positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;

	positionsGPU[i].x = std::fmaf(velocitiesGPU[i].x, 0.1f, positionsGPU[i].x);
	positionsGPU[i].y = std::fmaf(velocitiesGPU[i].y, 0.1f, positionsGPU[i].y);
	positionsGPU[i].z = std::fmaf(velocitiesGPU[i].z, 0.1f, positionsGPU[i].z);

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float* massesGPU, int n_particles)
{
	int nthreads = 32;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
