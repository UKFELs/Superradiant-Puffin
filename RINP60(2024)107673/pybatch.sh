#!/bin/bash
#BSUB -o stdout.pybatch2.txt
#BSUB -e stderr.pybatch2.txt
#BSUB -R "span[ptile=2]"
#BSUB -q scafellpikeSKL
#BSUB -n 8 
#BSUB -J pybatch
#BSUB -W 16:00

export py_script_directory=$HCBASE/plot-script

# setup modules
. /etc/profile.d/modules.sh
module load intel_mpi > /dev/null 2>&1
module load intel
module load python3/anaconda

export LD_LIBRARY_PATH=/lustre/scafellpike/local/HCP098/jkj01/pxp52-jkj01/hdf5/lib:$LD_LIBRARY_PATH
for i in {0..800}
do
   python3 $py_script_directory/plot1D_elec_field_batch.py $i SSS_e_ SSS_ap_ 238 246
done
mkdir img2
mv SSS_ap_*_elec.png img2
zip -r img2.zip img2

