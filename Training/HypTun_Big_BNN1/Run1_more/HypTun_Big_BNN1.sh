#!/bin/sh
#BSUB -J HypTun_Big_BNN1
#BSUB -q hpc
#BSUB -W 72:00
#BSUB -n 1
#BSUB -M 3GB
#BSUB -o Output_%J.out
#BSUB -e Error_%J.out



# here follow the commands you want to execute
set -e

module load python3/3.7.2
module load gcc/4.9.2
module load qt
source ~/specialeenv2/bin/activate

#Running code
python3 /zhome/13/e/97883/Documents/Speciale_code/Code/HypTun_Big_BNN1/Run1_more/HypTun_Big_BNN1_1.py
