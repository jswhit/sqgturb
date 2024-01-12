#!/bin/sh
#SBATCH --ntasks-per-node=40 --nodes=1
#SBATCH -t 2:00:00
#SBATCH -A gsienkf 
##SBATCH -q debug
#SBATCH -J sqg_run
#SBATCH -o sqg_run.out
#SBATCH -e sqg_run.err
source ~/bin/condapy
export OMP_NUM_THREADS=40
python sqg_run.py
