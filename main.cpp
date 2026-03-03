// =============================================================
//  main.cpp
//  Equivalent of the __main__ block in sqg.py.
//  Writes output to sqg.nc using the NetCDF-C library.
//
//  Compile:
//    g++ -I<NETCDF_AND_FFTW_INCDIR> -std=c++17 -O3 -march=native main.cpp -L<NETCDF_AND_FFTW_LIBDIR> -lfftw3f -lm -lnetcdf -o sqg
// =============================================================

#include "sqg.hpp"

#include <netcdf.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ------------------------------------------------------------------
//  Thin error-checking wrapper for NetCDF calls
// ------------------------------------------------------------------
static void nc_check(int status, const char* context = "")
{
    if (status != NC_NOERR) {
        std::string msg = "NetCDF error";
        if (context && context[0]) msg += std::string(" in ") + context;
        msg += ": ";
        msg += nc_strerror(status);
        throw std::runtime_error(msg);
    }
}

int main()
{
    // ---- model parameters ---------------------------------------
    const int   N          = 64;
    const float dt         = 1800.f;
    const float diff_efold = 24.f * 3600.f;
    const int   norder     = 8;
    const float r          = 0.f;
    const float nsq        = 1.e-4f;
    const float f          = 1.e-4f;
    const float g          = 9.8f;
    const float theta0     = 300.f;
    const float H          = 10.e3f;
    const float U          = 20.f;
    const float L          = 20.e6f;
    const float tdiab      = 10.f * 86400.f;
    const float scalefact  = f * theta0 / g;

    // ---- create initial PV --------------------------------------
    //std::mt19937 rng(42);
    //std::normal_distribution<float> gauss(0.f, 100.f);

    std::vector<float> pv(2 * N * N, 0);
    //for (auto& v : pv) v = gauss(rng);

    // add isolated blob on lid (k==1)
    const int nexp = 20;
    for (int iy = 0; iy < N; ++iy) {
        float y = float(iy) * 2.f * float(M_PI) / float(N);
        for (int ix = 0; ix < N; ++ix) {
            float x   = float(ix) * 2.f * float(M_PI) / float(N);
            float blob = 2000.f
                       * std::pow(std::sin(x / 2.f), 2 * nexp)
                       * std::pow(std::sin(y),            nexp);
            pv[1 * N * N + iy * N + ix] += blob;
        }
    }

    // remove area mean from each level
    for (int k = 0; k < 2; ++k) {
        double sum = 0.;
        for (int i = 0; i < N * N; ++i) sum += pv[k * N * N + i];
        float mean = float(sum / (N * N));
        for (int i = 0; i < N * N; ++i) pv[k * N * N + i] -= mean;
    }

    // ---- initialise model ---------------------------------------
    SQG model(pv, N, f, nsq, L, H, U, r, tdiab,
              norder, diff_efold, theta0, g, dt);

    const double outputinterval = 6. * 3600.;
    const double tmax           = 300. * 86400.;
    const int    nsteps         = int(tmax / outputinterval);
    const int    ntimesteps     = int(outputinterval / model.dt());

    std::cout << "SQG model: N=" << N
              << "  dt=" << dt << " s"
              << "  ntimesteps/output=" << ntimesteps
              << "\n";

    // ---- create NetCDF file -------------------------------------
    int ncid;
    nc_check(nc_create("sqg.nc", NC_CLOBBER | NC_NETCDF4, &ncid), "create");

    // global attributes
    float attr_r = model.r(0);
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "r",          NC_FLOAT, 1, &attr_r));
    float attr_f = model.f();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "f",          NC_FLOAT, 1, &attr_f));
    float attr_U = model.U();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "U",          NC_FLOAT, 1, &attr_U));
    float attr_L = model.L();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "L",          NC_FLOAT, 1, &attr_L));
    float attr_H = model.H();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "H",          NC_FLOAT, 1, &attr_H));
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "g",          NC_FLOAT, 1, &g));
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "theta0",     NC_FLOAT, 1, &theta0));
    float attr_nsq = model.nsq();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "nsq",        NC_FLOAT, 1, &attr_nsq));
    float attr_tdiab = model.tdiab();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "tdiab",      NC_FLOAT, 1, &attr_tdiab));
    float attr_dt = model.dt();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "dt",         NC_FLOAT, 1, &attr_dt));
    float attr_de = model.diff_efold();
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "diff_efold", NC_FLOAT, 1, &attr_de));
    int   attr_do = model.diff_order();
    nc_check(nc_put_att_int  (ncid, NC_GLOBAL, "diff_order", NC_INT,   1, &attr_do));

    // dimensions
    int dim_x, dim_y, dim_z, dim_t;
    nc_check(nc_def_dim(ncid, "x", N,       &dim_x));
    nc_check(nc_def_dim(ncid, "y", N,       &dim_y));
    nc_check(nc_def_dim(ncid, "z", 2,       &dim_z));
    nc_check(nc_def_dim(ncid, "t", NC_UNLIMITED, &dim_t));

    // coordinate variables
    int var_x, var_y, var_z, var_t;
    nc_check(nc_def_var(ncid, "x", NC_FLOAT, 1, &dim_x, &var_x));
    nc_check(nc_def_var(ncid, "y", NC_FLOAT, 1, &dim_y, &var_y));
    nc_check(nc_def_var(ncid, "z", NC_FLOAT, 1, &dim_z, &var_z));
    nc_check(nc_def_var(ncid, "t", NC_FLOAT, 1, &dim_t, &var_t));

    const char* m_str = "meters";
    const char* s_str = "seconds";
    nc_check(nc_put_att_text(ncid, var_x, "units", strlen(m_str), m_str));
    nc_check(nc_put_att_text(ncid, var_y, "units", strlen(m_str), m_str));
    nc_check(nc_put_att_text(ncid, var_z, "units", strlen(m_str), m_str));
    nc_check(nc_put_att_text(ncid, var_t, "units", strlen(s_str), s_str));

    // pv variable  (t, z, y, x) with compression
    int pv_dims[4] = {dim_t, dim_z, dim_y, dim_x};
    int var_pv;
    nc_check(nc_def_var(ncid, "pv", NC_FLOAT, 4, pv_dims, &var_pv));
    nc_check(nc_def_var_deflate(ncid, var_pv, /*shuffle=*/1, /*deflate=*/1, /*level=*/1));
    const char* K_str = "K";
    nc_check(nc_put_att_text(ncid, var_pv, "units", strlen(K_str), K_str));

    nc_check(nc_enddef(ncid));

    // write coordinate variables
    {
        std::vector<float> xc(N), yc(N);
        for (int i = 0; i < N; ++i) xc[i] = yc[i] = float(i) * L / float(N);
        nc_check(nc_put_var_float(ncid, var_x, xc.data()));
        nc_check(nc_put_var_float(ncid, var_y, yc.data()));
        float zc[2] = {0.f, H};
        nc_check(nc_put_var_float(ncid, var_z, zc));
    }

    // ---- time loop ----------------------------------------------
    int nout = 0;
    while (model.t() < tmax) {
        auto pv_out = model.advance(ntimesteps);
        double t_now = model.t();

        // diagnostics
        float pvmin = *std::min_element(pv_out.begin(), pv_out.end());
        float pvmax = *std::max_element(pv_out.begin(), pv_out.end());
        std::printf("hr=%8.2f  min/max pv  %10.4f  %10.4f\n",
                    t_now / 3600., scalefact * pvmin, scalefact * pvmax);

        // write t
        size_t t_idx = size_t(nout);
        float  t_val = float(t_now);
        nc_check(nc_put_var1_float(ncid, var_t, &t_idx, &t_val));

        // write pv  (shape: 1 × 2 × N × N)
        size_t start[4] = {size_t(nout), 0, 0, 0};
        size_t count[4] = {1, 2, size_t(N), size_t(N)};
        nc_check(nc_put_vara_float(ncid, var_pv, start, count, pv_out.data()));

        nc_check(nc_sync(ncid));
        ++nout;
    }

    nc_check(nc_close(ncid));
    std::cout << "Wrote " << nout << " frames to sqg.nc\n";
    return 0;
}
