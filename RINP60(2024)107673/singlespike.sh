#!/bin/bash
#BSUB -o stdout.sss.txt
#BSUB -e stderr.sss.txt
#BSUB -R "span[ptile=2]"
#BSUB -q scafellpikeSKL
#BSUB -n 8 
#BSUB -J 4pirho-1
#BSUB -W 16:00

export PUFFDIR=$HCBASE/puffin/bin
export MYSCRIPT=$HCBASE/rafel-script
export MYHOME=`pwd`
export OMP_NUM_THREADS=1
export OPC_HOME=$HCBASE/OPC/Physics-OPC-0.7.10.3
export RAFEL=$HCBASE/OPC/Physics-OPC-0.7.10.3/opc-puffin
export py_script_directory=$HCBASE/plot-script

# setup simulation parameters
# basename of the input file
BASENAME=SSS
LFN=41

# setup modules
. /etc/profile.d/modules.sh
module load intel_mpi > /dev/null 2>&1
module load intel
module load python3/anaconda

export LD_LIBRARY_PATH=/lustre/scafellpike/local/HCP098/jkj01/pxp52-jkj01/hdf5/lib:$LD_LIBRARY_PATH

ploop=0
echo "Running loop number: ${ploop}"
mpiexec.hydra -np 64 $PUFFDIR/puffin ${BASENAME}.in
mv ${BASENAME}_electrons_${LFN}.h5 beam_file.h5
python3 $MYSCRIPT/resetElectrons_h5_v3_fix.py beam_file.h5 beam_spread_${ploop}.png 238
mv ${BASENAME}_aperp_${LFN}.h5 seed_file.h5
python3 $MYSCRIPT/editFields_h5_v2.py seed_file.h5 seed_file_${ploop}.png 238 262
cp beam_file.h5 beam_file_${ploop}.h5
cp seed_file.h5 seed_file_${ploop}.h5

for i in {0..40}; do mv ${BASENAME}_aperp_${i}.h5 ${BASENAME}_ap_$((40*${ploop}+${i})).h5; done
for i in {0..40}; do mv ${BASENAME}_electrons_${i}.h5 ${BASENAME}_e_$((40*${ploop}+${i})).h5; done
for i in {0..40}; do mv ${BASENAME}_integrated_${i}.h5 ${BASENAME}_int_$((40*${ploop}+${i})).h5; done

for loop in {1..800}
do
	echo "Running loop number: ${loop}"
	mpiexec.hydra -np 64 $PUFFDIR/puffin ${BASENAME}.ins
        mv ${BASENAME}_electrons_${LFN}.h5 beam_file.h5
        python3 $MYSCRIPT/resetElectrons_h5_v3_fix.py beam_file.h5 beam_spread_${loop}.png 238
        mv ${BASENAME}_aperp_${LFN}.h5 seed_file.h5
        python3 $MYSCRIPT/editFields_h5_v2.py seed_file.h5 seed_file_${loop}.png 238 262
        cp beam_file.h5 beam_file_${loop}.h5
        cp seed_file.h5 seed_file_${loop}.h5

        for i in {1..40}; do mv ${BASENAME}_aperp_${i}.h5 ${BASENAME}_ap_$((40*${loop}+${i})).h5; done
        for i in {1..40}; do mv ${BASENAME}_electrons_${i}.h5 ${BASENAME}_e_$((40*${loop}+${i})).h5; done
        for i in {1..40}; do mv ${BASENAME}_integrated_${i}.h5 ${BASENAME}_int_$((40*${loop}+${i})).h5; done
done
