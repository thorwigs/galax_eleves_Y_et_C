#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < n_particles; i++)
        {
            for (int j = 0; j < n_particles; j++)
            {
                if (i!=j) {
                    const float diffx = particles.x[j] - particles.x[i];
                    const float diffy = particles.y[j] - particles.y[i];
                    const float diffz = particles.z[j] - particles.z[i];

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];

                }
                
            }
        }
    }

    // #pragma omp parallel for
	// for (int i = 0; i < n_particles; i++)
	// {
	// 	velocitiesx[i] += accelerationsx[i] * 2.0f;
	// 	velocitiesy[i] += accelerationsy[i] * 2.0f;
	// 	velocitiesz[i] += accelerationsz[i] * 2.0f;
	// 	particles.x[i] += velocitiesx   [i] * 0.1f;
	// 	particles.y[i] += velocitiesy   [i] * 0.1f;
	// 	particles.z[i] += velocitiesz   [i] * 0.1f;
	// }



// OMP + xsimd version
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
              b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
              b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
              b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
              b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
              b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
              b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        rvelx_i = xs::fma(raccx_i, (b_type) 2.0f, rvelx_i);
        rvely_i = xs::fma(raccy_i, (b_type) 2.0f, rvely_i);
        rvelz_i = xs::fma(raccz_i, (b_type) 2.0f, rvelz_i);

        rposx_i.store_unaligned(xs::fma(rvelx_i, (b_type) 1.0f, rposx_i));
        rposy_i.store_unaligned(xs::fma(rvely_i, (b_type) 1.0f, rposy_i));
        rposz_i.store_unaligned(xs::fma(rvelz_i, (b_type) 1.0f, rposz_i));
    }

}

#endif // GALAX_MODEL_CPU_FAST
