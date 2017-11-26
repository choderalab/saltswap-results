#!/bin/bash

NPERT=(250 500 750 1000 1250 1500 1750 2000)

for U in ${NPERT[*]}
do
    mkdir "npert_${U}"
    cd "npert_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
