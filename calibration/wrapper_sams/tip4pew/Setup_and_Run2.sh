#!/bin/bash

REPEAT=(6 7 8 9 10)

for R in ${REPEAT[*]}
do
    sed "s/REPLACE/$R/" submit_template2.lsf > submit${R}.lsf
    bsub < submit${R}.lsf
done
