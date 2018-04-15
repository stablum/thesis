#!/bin/bash --login
#$ -l h_rt=120:00:00

#### ignored option -l gpu=GTX480
#### other possibility: -l gpu=GTX680
#### other possibility: -l fat,gpu=K20
#### other possibility: -l gpu=C2050
#### other possibility: -l gpu=GTX480

module add python/3.3.2
module add python/default
module add git/1.8.3.4
module add opencl-nvidia/5.5
module add cuda55/blas/5.5.22
module add cuda55/fft/5.5.22
module add cuda55/profiler/5.5.22
module add cuda55/tdk/5.319.43
module add cuda55/toolkit/5.5.22
source ~/venv2/bin/activate
cd thesis
nvidia-smi
echo -n "hostname:"
hostname
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fstablum/.local/lib
python3 $@

