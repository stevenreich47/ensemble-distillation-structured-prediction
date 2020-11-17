#! /usr/bin/env bash

for MODEL in 0 1 2 3 4 5 6 7 8; do

qsub -q gpu.q@@2080 -l gpu=1,num_proc=12,mem_free=20G,h_rt=40:00:00 -cwd train_teacher.sh iid ${MODEL}

done

