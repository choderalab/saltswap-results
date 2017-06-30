#!/bin/bash

REPEAT=(1 2 3 4 5)

for R in ${REPEAT[*]}
do
    sed "s/REPLACE/$R/" submit_template.lsf > submit${R}.lsf
    bsub < submit${R}.lsf
done
