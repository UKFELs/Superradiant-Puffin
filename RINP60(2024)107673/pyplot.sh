#!/bin/bash
#BSUB -o stdout.pyplot_new1.txt
#BSUB -e stderr.pyplot_new1.txt
#BSUB -R "span[ptile=2]"
#BSUB -q scafellpikeSKL
#BSUB -n 8 
#BSUB -J pyplot
#BSUB -W 16:00

export py_script_directory=$HCBASE/plot-script

# setup modules
. /etc/profile.d/modules.sh
module load intel_mpi > /dev/null 2>&1
module load intel
module load python3/anaconda

export LD_LIBRARY_PATH=/lustre/scafellpike/local/HCP098/jkj01/pxp52-jkj01/hdf5/lib:$LD_LIBRARY_PATH

python3 $py_script_directory/read_aperp_fwhm_h5_bash.py SSS_ap_ 0 32040 40 238 247
