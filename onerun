#!/bin/bash
#BSUB -n 1
#BSUB -R "span[hosts=1]"

module add seas-lab-vorobeychik

n=$1
m=$2
t=$3
SEED=$4
RANK=$5
TYPE=$6
FACTOR=$7
# python utilities/generate_data.py -n ${n} -m ${m} -t ${t} -s ${SEED} -r ${RANK}
if [[ $TYPE == "both_prior" ]]; then
    python simulation.py -n ${n} -m ${m} -t ${t} -s ${SEED} -r ${RANK} -k ${TYPE} -f ${FACTOR}
elif [[ $TYPE == "DSEM" ]]; then
    Rscript --vanilla utilities/simulation_DSEM.R ${n} ${m} ${t} ${RANK} ${SEED} ${TYPE}
elif [[ $TYPE == "TVAR" ]]; then
    Rscript --vanilla utilities/simulation_VAR.R ${n} ${m} ${t} ${RANK} ${SEED} ${TYPE}
elif [[ $TYPE == *"_"* ]]; then
    Rscript --vanilla utilities/simulation_MIRT.R ${n} ${m} ${t} ${RANK} ${SEED} ${TYPE}
else
    python simulation.py -n ${n} -m ${m} -t ${t} -s ${SEED} -r ${RANK} -k ${TYPE} -f ${FACTOR}
fi