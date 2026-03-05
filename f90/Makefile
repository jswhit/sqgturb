# Makefile for sqg_mpi (Fortran 95 serial SQG)
# Adjust FC, FFTW_DIR, NETCDF_DIR to match your installation.

FC       = gfortran
FFLAGS   = -std=f2003 -O3 -march=native -ffree-line-length-none

# FFTW3 single-precision
FFTW_INC = -I/Users/jwhitaker/miniconda3/envs/netcdf4/include
FFTW_LIB = -L/Users/jwhitaker/miniconda3/envs/netcdf4/lib -lfftw3f

# NetCDF-Fortran  (nf90_* interface)
NC_INC   = -I/Users/jwhitaker/miniconda3/envs/netcdf4/include
NC_LIB   = -L/Users/jwhitaker/miniconda3/envs/netcdf4/lib -lnetcdff -lnetcdf

LDFLAGS  = $(FFTW_LIB) $(NC_LIB) -lm

TARGET   = sqg
OBJS     = sqg_mod.o sqg_main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^ $(LDFLAGS)

sqg_mod.o: sqg_mod.f90
	$(FC) $(FFLAGS) $(FFTW_INC) $(NC_INC) -c $<

sqg_main.o: sqg_main.f90 sqg_mod.o
	$(FC) $(FFLAGS) $(FFTW_INC) $(NC_INC) -c $<

clean:
	rm -f $(TARGET) $(OBJS) *.mod

.PHONY: all clean
