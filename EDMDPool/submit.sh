#!/bin/sh
#JSUB -m gpu01
#JSUB -q normal
#JSUB -n 4
#JSUB -e ./result/error.%J
#JSUB -o ./result/output.%J
#JSUB -J andi
./run_EDMDPool.sh  PROTEINS 0 0
