#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --cpus-per-task=50
#SBATCH --mem=70G

#SBATCH -t 90:00:00  
#SBATCH -p longq

cd ./scripts/m_choffe_HAN/

source activate ../CondaCloneWrkFlow

python Hyperopt_HAN_generator.py