#!/bin/sh
#BSUB -J Bay118bus
#BSUB -q hpc
#BSUB -W 72:00
#BSUB -n 1
#BSUB -M 4GB
#BSUB -o Output_1.out
#BSUB -e Error_1.out



# here follow the commands you want to execute
set -e

module load python3/3.7.2
module load gcc/4.9.2
module load qt
source ~/specialeenv2/bin/activate

#Running code
python3 /zhome/13/e/97883/Documents/Speciale_code/Code/Bay118bus/Run3/Bay118bus_1.py
