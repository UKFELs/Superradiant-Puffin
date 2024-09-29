#!/bin/bash
#BSUB -o stdout.pybatch2periods3.txt
#BSUB -e stderr.pybatch2periods3.txt
#BSUB -R "span[ptile=2]"
#BSUB -q scafellpikeSKL
#BSUB -n 8 
#BSUB -J pybatch2periods
#BSUB -W 16:00

export py_script_directory=$HCBASE/plot-script

# setup modules
. /etc/profile.d/modules.sh
module load intel_mpi > /dev/null 2>&1
module load intel
module load python3/anaconda

export LD_LIBRARY_PATH=/lustre/scafellpike/local/HCP098/jkj01/pxp52-jkj01/hdf5/lib:$LD_LIBRARY_PATH

python3 $py_script_directory/plot1D_elec_field_batch_2periods.py 100 SSS_e_ SSS_ap_ 238 246

mkdir img2periods3
mv SSS_ap_*_elec.png img2periods3
zip -r img2periods3.zip img2periods3

