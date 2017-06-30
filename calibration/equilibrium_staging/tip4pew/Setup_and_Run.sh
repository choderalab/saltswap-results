#!/bin/bash

POTENTIAL=(313.18 313.58 314.00 314.66 315.78 316.91 317.93)

for U in ${POTENTIAL[*]}
do
    mkdir "deltamu_${U}"
    cd "deltamu_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
