/**
 * sqg_run.cpp
 *
 * Example driver for the C++ SQG model, mirroring sqg_run.py.
 * Runs the model and prints hourly statistics to stdout.
 *
 * Build with CMake:
 *   mkdir build && cd build && cmake .. && make -j
 *   ./sqg_run
 */
#include "sqg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main()
{
    // ------------------------------------------------------------------
    // Model parameters (match sqg_run.py defaults)
    // ------------------------------------------------------------------
    const int   N          = 96;
    const float dt         = 900.0f;          // seconds
    const float diff_efold = 86400.0f / 2.0f; // half a day
    const int   norder     = 8;
    const bool  dealias    = true;

    const float dek    = 0.0f;                // Ekman depth (zero → no Ekman)
    const float nsq    = 1.0e-4f;
    const float f      = 1.0e-4f;
    const float g      = 9.8f;
    const float theta0 = 300.0f;
    const float H      = 10.0e3f;
    const float r      = dek * nsq / f;
    const float U      = 20.0f;              // jet speed (m/s)
    const float Lr     = std::sqrt(nsq) * H / f;
    const float L      = 20.0f * Lr;
    const float tdiab  = 10.0f * 86400.0f;
    const bool  sym    = true;

    const float scalefact = f * theta0 / g;

    const double tmax           = 300.0 * 86400.0;
    const double output_interval = 6.0  * 3600.0;

    // ------------------------------------------------------------------
    // Initialise PV: random noise + isolated blob on the lid
    // ------------------------------------------------------------------
    std::vector<float> pv(2 * N * N, 0.0f);

    std::mt19937                    rng(42);
    std::normal_distribution<float> noise(0.0f, 100.0f);

    for (int i = 0; i < 2 * N * N; i++)
        pv[i] = noise(rng);

    // Add isolated blob on lid (level 1)
    const int nexp = 20;
    for (int iy = 0; iy < N; iy++) {
        float y = 2.0f * static_cast<float>(M_PI) * iy / N;
        for (int ix = 0; ix < N; ix++) {
            float x   = 2.0f * static_cast<float>(M_PI) * ix / N;
            float blob = 2000.0f
                       * std::pow(std::sin(x / 2.0f), 2 * nexp)
                       * std::pow(std::sin(y),          nexp);
            pv[1 * N * N + iy * N + ix] += blob;
        }
    }

    // Remove area mean from each level
    for (int lev = 0; lev < 2; lev++) {
        float sum = 0.0f;
        for (int k = 0; k < N * N; k++) sum += pv[lev * N * N + k];
        float mean = sum / static_cast<float>(N * N);
        for (int k = 0; k < N * N; k++) pv[lev * N * N + k] -= mean;
    }

    // ------------------------------------------------------------------
    // Construct model
    // ------------------------------------------------------------------
    SQG model(pv.data(), N,
              f, nsq, L, H, U, r, tdiab,
              norder, diff_efold,
              theta0, g,
              sym,
              dt,
              dealias,
              /*threads=*/1,
              /*tstart=*/0.0f);

    // Set number of sub-steps per advance() call
    model.timesteps = static_cast<int>(std::round(output_interval / dt));

    // ------------------------------------------------------------------
    // Time loop
    // ------------------------------------------------------------------
    std::vector<float> pv_out(2 * N * N);

    std::printf("%-12s  %-14s  %-14s  %-14s\n",
                "time(hr)", "max_speed", "pv_min(K)", "pv_max(K)");

    while (model.t < tmax) {
        model.advance(pv_out.data());

        double hr = model.t / 3600.0;

        // Max wind speed (level 1 = lid)
        float spd_max = 0.0f;
        for (int k = 0; k < N * N; k++) {
            float uu = model.u[N * N + k];
            float vv = model.v[N * N + k];
            float spd = std::sqrt(uu * uu + vv * vv);
            if (spd > spd_max) spd_max = spd;
        }

        // PV extrema on lid (level 1), converted to temperature units (K)
        float pv_min =  1.0e30f;
        float pv_max = -1.0e30f;
        for (int k = 0; k < N * N; k++) {
            float val = scalefact * pv_out[N * N + k];
            if (val < pv_min) pv_min = val;
            if (val > pv_max) pv_max = val;
        }

        std::printf("%-12.2f  %-14.4f  %-14.4f  %-14.4f\n",
                    hr, spd_max, pv_min, pv_max);
        std::fflush(stdout);
    }

    return 0;
}
