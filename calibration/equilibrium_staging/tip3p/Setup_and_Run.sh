#!/bin/bash

POTENTIAL=(314.85 315.24 315.68 316.39 317.61 318.78 319.83)

for U in ${POTENTIAL[*]}
do
    mkdir "deltamu_${U}"
    cd "deltamu_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
