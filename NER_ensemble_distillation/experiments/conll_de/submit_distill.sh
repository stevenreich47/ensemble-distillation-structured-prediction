#! /usr/bin/env bash

for TEACHERS in 3 5 8; do

qsub -q gpu.q@@2080 -l gpu=1,num_proc=12,mem_free=10G,h_rt=40:00:00 -cwd distill_ensemble.sh iid ${TEACHERS}

done

