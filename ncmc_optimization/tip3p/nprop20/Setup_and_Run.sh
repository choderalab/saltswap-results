#!/bin/bash

NPERT=(125 250 375 500 625 750 875 1000)

for U in ${NPERT[*]}
do
    mkdir "npert_${U}"
    cd "npert_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
