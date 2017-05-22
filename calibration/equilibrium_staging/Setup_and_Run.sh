#!/bin/bash

POTENTIAL=(315.0 317.5 320.0 322.5 325.0 327.5 330.0 332.5 335.0 337.5 340.0 342.5 345.0)

for U in ${POTENTIAL[*]}
do
    mkdir "deltamu_${U}"
    cd "deltamu_${U}"
    sed "s/REPLACE/$U/" ../submit_template.lsf > submit.lsf
    bsub < submit.lsf
    cd ../
done
