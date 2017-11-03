#!/bin/bash

REPEAT=(1 2 3)

for R in ${REPEAT[*]}
do
    sed "s/REPLACE/$R/" submit_template.lsf > submit${R}.lsf
    bsub < submit${R}.lsf
done
