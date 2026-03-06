# Makefile for the C SQG model
# Adjust CC and library paths for your system.

CC      = gcc
CFLAGS  = -std=c99 -O3 -march=native -Wall -Wextra

# FFTW3 single-precision
FFTW_INC = -I/Users/jwhitaker/miniconda3/envs/netcdf4/include
FFTW_LIB = -L/Users/jwhitaker/miniconda3/envs/netcdf4/lib -lfftw3f

# NetCDF
NC_INC   = -I/Users/jwhitaker/miniconda3/envs/netcdf4/include
NC_LIB   = -L/Users/jwhitaker/miniconda3/envs/netcdf4/lib -lnetcdf

LDFLAGS  = $(FFTW_LIB) $(NC_LIB) -lm

TARGET  = sqg
OBJS    = sqg.o sqg_main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sqg.o: sqg.c sqg.h
	$(CC) $(CFLAGS) $(FFTW_INC) $(NC_INC) -c $<

sqg_main.o: sqg_main.c sqg.h
	$(CC) $(CFLAGS) $(FFTW_INC) $(NC_INC) -c $<

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
