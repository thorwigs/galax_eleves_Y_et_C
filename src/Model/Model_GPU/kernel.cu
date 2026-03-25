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
			sh_positions[kp] = __ldg(&positionsGPU[j+kp]);
			// sh_positions[kp] = positionsGPU[j+kp];
		__syncthreads();


		if (i<n_particles) {

			
			// for (int k = 0; k < TAILLE && j+k < n_particles; k++)
			#pragma unroll 32
			for (int k = 0; (k < blockDim.x) && ((j+k) < n_particles); k++)
			{
				float4 pj = sh_positions[k];

				float4 diff = make_float4(
				pj.x - pos_i.x,
				pj.y - pos_i.y,
				pj.z - pos_i.z,
				0.0f
			);
			
				// const float diffx = sh_positions[k].x - pos_i.x;
				// const float diffy = sh_positions[k].y - pos_i.y;
				// const float diffz = sh_positions[k].z - pos_i.z;

				// const float diffx = positionsGPU[j].x - pos_i.x;
				// const float diffy = positionsGPU[j].y - pos_i.y;
				// const float diffz = positionsGPU[j].z - pos_i.z;

				//float dij = fmaf(diff.x, diff.x, fmaf(diff.y, diff.y, fmaf(diff.z, diff.z, 0.0f)));
				float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;


				// if (dij < 1.0)
				// {
				// 	dij = 10.0;
				// }
				// else
				// {
				// 	dij = rsqrtf(dij);
				// 	dij = 10.0*(dij*dij*dij);
				// }

				dij = rsqrtf(dij);
				dij = 10*(dij * dij * dij);
				dij = fminf(10, dij);

				// float dij = fmaf(diff.x, diff.x, fmaf(diff.y, diff.y, fmaf(diff.z, diff.z, EPS)));
				// float inv = rsqrtf(dij);
				// dij = 10.0f * inv * inv * inv;

				acc.x += diff.x * dij * sh_positions[k].w;
				acc.y += diff.y * dij * sh_positions[k].w;
				acc.z += diff.z * dij * sh_positions[k].w;
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
	int nthreads = 1024;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads, nthreads*sizeof(float4)>>>(positionsGPU, velocitiesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
