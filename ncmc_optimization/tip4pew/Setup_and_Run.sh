#!/bin/bash

NPERT=(1 2500 5000 7500 10000 12500 15000 17500 20000)

for U in ${NPERT[*]}
do
    mkdir "npert_${U}"
    cd "npert_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
