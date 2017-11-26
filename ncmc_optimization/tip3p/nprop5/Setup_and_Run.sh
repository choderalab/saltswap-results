#!/bin/bash

NPERT=(500 1000 1500 2000 2500 3000 3500 4000)

for U in ${NPERT[*]}
do
    mkdir "npert_${U}"
    cd "npert_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
