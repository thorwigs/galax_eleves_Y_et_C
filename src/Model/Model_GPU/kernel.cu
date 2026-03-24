#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#include <cmath>
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	float4 pos_i;
	float3 acc;

	if (i < n_particles) {

		pos_i = positionsGPU[i];
		acc = {0.0f, 0.0f, 0.0f};
	}


	extern __shared__ float4 sh_positions[];


	for (int j = 0; j < n_particles; j=j+blockDim.x)
	{

		__syncthreads();

		int kp = threadIdx.x;
		if (j+kp < n_particles)
			sh_positions[kp] = positionsGPU[j+kp];
		__syncthreads();

		if (i<n_particles) {
			// for (int k = 0; k < TAILLE && j+k < n_particles; k++)
			for (int k = 0; (k < blockDim.x) && ((j+k) < n_particles); k++)
			{
				const float diffx = sh_positions[k].x - pos_i.x;
				const float diffy = sh_positions[k].y - pos_i.y;
				const float diffz = sh_positions[k].z - pos_i.z;

				// const float diffx = positionsGPU[j].x - pos_i.x;
				// const float diffy = positionsGPU[j].y - pos_i.y;
				// const float diffz = positionsGPU[j].z - pos_i.z;

				float dij = fmaf(diffx, diffx, fmaf(diffy, diffy, fmaf(diffz, diffz, 0.0f)));

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = rsqrtf(dij);
					dij = (10.0 * dij) * (dij * dij);
				}
				// dij = rsqrtf(dij);
				// dij = 10 * (dij * dij * dij);
				// dij = fminf(10, dij);

				acc.x += diffx * dij * sh_positions[k].w;
				acc.y += diffy * dij * sh_positions[k].w;
				acc.z += diffz * dij * sh_positions[k].w;
			}
		}
	}

	if (i < n_particles) {
		velocitiesGPU[i].x += acc.x * 2.0f;
		velocitiesGPU[i].y += acc.y * 2.0f;
		velocitiesGPU[i].z += acc.z * 2.0f;
	}
}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= n_particles) {
		return;
	}

	positionsGPU[i].x = std::fmaf(velocitiesGPU[i].x, 0.1f, positionsGPU[i].x);
	positionsGPU[i].y = std::fmaf(velocitiesGPU[i].y, 0.1f, positionsGPU[i].y);
	positionsGPU[i].z = std::fmaf(velocitiesGPU[i].z, 0.1f, positionsGPU[i].z);

}

void update_position_cu(float4* positionsGPU, float3* velocitiesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads, nthreads*sizeof(float4)>>>(positionsGPU, velocitiesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
