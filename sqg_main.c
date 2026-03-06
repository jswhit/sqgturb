/*
 * sqg_main.c  -  Driver for the C SQG model
 * Equivalent of the C++ __main__ / main.cpp.
 *
 * Compile:
 *   gcc -std=c99 -O3 -march=native \
 *       sqg.c sqg_main.c -lfftw3f -lnetcdf -lm -o sqg
 * Run:
 *   ./sqg
 */

#include "sqg.h"

#include <math.h>
#include <netcdf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  NetCDF error helper                                                */
/* ------------------------------------------------------------------ */
static void nc_check(int rc, const char *ctx)
{
    if (rc != NC_NOERR) {
        fprintf(stderr, "NetCDF error [%s]: %s\n", ctx, nc_strerror(rc));
        exit(1);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    /* ---- model parameters ----------------------------------- */
    const int    N          = 64;
    const real_t dt_val     = 1800.0f;
    const real_t diff_efold = 24.0f * 3600.0f;
    const int    norder     = 8;
    const real_t r_val      = 0.0f;
    const real_t nsq        = 1.0e-4f;
    const real_t f_val      = 1.0e-4f;
    const real_t g_val      = 9.8f;
    const real_t theta0     = 300.0f;
    const real_t H_val      = 10.0e3f;
    const real_t U_val      = 20.0f;
    const real_t L_val      = 20.0e6f;
    const real_t tdiab      = 10.0f * 86400.0f;
    const real_t scalefact  = f_val * theta0 / g_val;

    const double outputinterval = 6.0 * 3600.0;
    const double tmax           = 300.0 * 86400.0;

    const real_t pi2 = 2.0f * (real_t)M_PI;

    /* ---- allocate initial PV -------------------------------- */
    real_t *pv     = (real_t *)calloc((size_t)(2 * N * N), sizeof(real_t));
    real_t *pv_out = (real_t *)malloc((size_t)(2 * N * N) * sizeof(real_t));
    if (!pv || !pv_out) {
        fprintf(stderr, "main: malloc failed\n");
        return 1;
    }

    /* Blob on lid (component k=1): 2000*(sin(x/2)^20)*(sin(y)^20) */
    for (int iy = 0; iy < N; ++iy) {
        real_t y_val = (real_t)iy * pi2 / (real_t)N;
        for (int ix = 0; ix < N; ++ix) {
            real_t x_val = (real_t)ix * pi2 / (real_t)N;
            real_t blob  = 2000.0f
                         * powf(sinf(0.5f * x_val), 20)
                         * powf(sinf(y_val),         20);
            /* C++ layout: pv[k*N*N + j*N + i] with j=iy, i=ix */
            pv[1 * N * N + iy * N + ix] += blob;
        }
    }

    /* Remove area mean from each component */
    for (int k = 0; k < 2; ++k) {
        double sum = 0.0;
        for (int i = 0; i < N * N; ++i)
            sum += pv[k * N * N + i];
        real_t mean = (real_t)(sum / (N * N));
        for (int i = 0; i < N * N; ++i)
            pv[k * N * N + i] -= mean;
    }

    /* ---- initialise model ----------------------------------- */
    SQG *model = sqg_create(pv, N,
                             f_val, nsq, L_val, H_val, U_val, r_val, tdiab,
                             norder, diff_efold, theta0, g_val, dt_val, 0.0);
    if (!model) {
        fprintf(stderr, "sqg_create failed\n");
        free(pv); free(pv_out);
        return 1;
    }

    int ntimesteps = (int)(outputinterval / (double)model->dt);
    printf("SQG C model: N=%d  dt=%.0f s  ntimesteps=%d\n",
           model->N, (double)model->dt, ntimesteps);

    /* ---- create NetCDF output file -------------------------- */
    int ncid;
    nc_check(nc_create("sqg.nc", NC_CLOBBER | NC_NETCDF4, &ncid), "create");

    /* global attributes */
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "r",
                              NC_FLOAT, 1, &model->r[0]),          "att r");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "f",
                              NC_FLOAT, 1, &model->f),             "att f");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "U",
                              NC_FLOAT, 1, &model->U),             "att U");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "L",
                              NC_FLOAT, 1, &model->L),             "att L");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "H",
                              NC_FLOAT, 1, &model->H),             "att H");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "g",
                              NC_FLOAT, 1, &g_val),                "att g");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "theta0",
                              NC_FLOAT, 1, &theta0),               "att theta0");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "nsq",
                              NC_FLOAT, 1, &model->nsq),           "att nsq");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "tdiab",
                              NC_FLOAT, 1, &model->tdiab),         "att tdiab");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "dt",
                              NC_FLOAT, 1, &model->dt),            "att dt");
    nc_check(nc_put_att_float(ncid, NC_GLOBAL, "diff_efold",
                              NC_FLOAT, 1, &model->diff_efold),    "att diff_efold");
    nc_check(nc_put_att_int  (ncid, NC_GLOBAL, "diff_order",
                              NC_INT,   1, &model->diff_order),    "att diff_order");

    /* dimensions */
    int dim_x, dim_y, dim_z, dim_t;
    nc_check(nc_def_dim(ncid, "x", (size_t)N,       &dim_x), "dim x");
    nc_check(nc_def_dim(ncid, "y", (size_t)N,       &dim_y), "dim y");
    nc_check(nc_def_dim(ncid, "z", 2,               &dim_z), "dim z");
    nc_check(nc_def_dim(ncid, "t", NC_UNLIMITED,    &dim_t), "dim t");

    /* coordinate variables */
    int var_x, var_y, var_z, var_t;
    nc_check(nc_def_var(ncid, "x", NC_FLOAT, 1, &dim_x, &var_x), "def x");
    nc_check(nc_def_var(ncid, "y", NC_FLOAT, 1, &dim_y, &var_y), "def y");
    nc_check(nc_def_var(ncid, "z", NC_FLOAT, 1, &dim_z, &var_z), "def z");
    nc_check(nc_def_var(ncid, "t", NC_FLOAT, 1, &dim_t, &var_t), "def t");
    nc_check(nc_put_att_text(ncid, var_x, "units", 6,  "meters"),  "att x units");
    nc_check(nc_put_att_text(ncid, var_y, "units", 6,  "meters"),  "att y units");
    nc_check(nc_put_att_text(ncid, var_z, "units", 6,  "meters"),  "att z units");
    nc_check(nc_put_att_text(ncid, var_t, "units", 7,  "seconds"), "att t units");

    /* pv variable: (t, z, y, x) in C / row-major order */
    int pv_dims[4] = { dim_t, dim_z, dim_y, dim_x };
    int var_pv;
    nc_check(nc_def_var(ncid, "pv", NC_FLOAT, 4, pv_dims, &var_pv), "def pv");
    nc_check(nc_def_var_deflate(ncid, var_pv, 1, 1, 1),             "deflate pv");
    nc_check(nc_put_att_text(ncid, var_pv, "units", 1, "K"),        "att pv units");

    nc_check(nc_enddef(ncid), "enddef");

    /* write coordinates */
    {
        real_t *xc = (real_t *)malloc((size_t)N * sizeof(real_t));
        real_t *yc = (real_t *)malloc((size_t)N * sizeof(real_t));
        for (int i = 0; i < N; ++i)
            xc[i] = yc[i] = (real_t)i * model->L / (real_t)N;
        nc_check(nc_put_var_float(ncid, var_x, xc), "put x");
        nc_check(nc_put_var_float(ncid, var_y, yc), "put y");
        free(xc); free(yc);

        real_t zc[2] = { 0.0f, model->H };
        nc_check(nc_put_var_float(ncid, var_z, zc), "put z");
    }

    /* ---- time loop ------------------------------------------ */
    int nout = 0;
    while (model->t < tmax) {
        sqg_advance(model, ntimesteps, NULL, pv_out);

        /* find min/max PV */
        real_t pvmin = pv_out[0], pvmax = pv_out[0];
        for (int i = 1; i < 2 * N * N; ++i) {
            if (pv_out[i] < pvmin) pvmin = pv_out[i];
            if (pv_out[i] > pvmax) pvmax = pv_out[i];
        }
        printf("hr=%8.2f  min/max pv  %10.4f  %10.4f\n",
               model->t / 3600.0,
               (double)(scalefact * pvmin),
               (double)(scalefact * pvmax));

        /* write pv record */
        size_t start[4] = { (size_t)nout, 0, 0, 0 };
        size_t count[4] = { 1, 2, (size_t)N, (size_t)N };
        nc_check(nc_put_vara_float(ncid, var_pv, start, count, pv_out),
                 "put pv");

        float tval = (float)model->t;
        size_t tidx = (size_t)nout;
        nc_check(nc_put_var1_float(ncid, var_t, &tidx, &tval), "put t");
        nc_check(nc_sync(ncid), "sync");

        ++nout;
    }

    nc_check(nc_close(ncid), "close");
    sqg_destroy(model);
    free(pv);
    free(pv_out);
    return 0;
}
