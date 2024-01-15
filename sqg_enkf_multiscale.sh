#!/bin/sh
##SBATCH -q debug
#SBATCH -t 08:00:00
#SBATCH -A gsienkf
#SBATCH -N 1  
#SBATCH --ntasks-per-node=40
#SBATCH -p orion
#SBATCH -J sqg_z2loc64_6hrly_rtps0p4_4800_2400_10
#SBATCH -e sqg_z2loc64_6hrly_rtps0p4_4800_2400_10.err
#SBATCH -o sqg_z2loc64_6hrly_rtps0p4_4800_2400_10.out
#source ~/bin/condapy
export OMP_NUM_THREADS=40
export PYTHONUNBUFFERED=1
export exptname="z2loc64_6hrly_rtps0p4_4800_2400_10"
python sqg_enkf_multiscale.py "[4800.e3,2400.e6]" "[10]"  0.4
