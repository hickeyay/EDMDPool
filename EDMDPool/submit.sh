#!/bin/sh
#JSUB -m gpu01
#JSUB -q normal
#JSUB -n 4
#JSUB -e ./result/error.%J
#JSUB -o ./result/output.%J
#JSUB -J andi
./run_EDMDPool.sh  PROTEINS 0 0


:<<!
    ./run_EDMDPool.sh DATA FOLD GPU
to run on dataset using fold number (1-10).
You can run
    ./run_EDMDPool.sh DD 0 0
to run on DD dataset with 10-fold cross validation on GPU #0.
    ./run_EDMDPool.sh COLLAB 1 1
to run on COLLAB dataset with first fold on GPU #1.
!
