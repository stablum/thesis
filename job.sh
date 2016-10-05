#!/bin/bash --login
#$ -l gpu=C2050
#$ -l h_rt=24:00:00
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
cd reimplementations
nvidia-smi
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fstablum/.local/lib
THEANO_FLAGS=mode=FAST_RUN,device=gpu,init_gpu_device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES,print_active_device=True,enable_initial_driver_test=True,warn_float64=raise,force_device=True,assert_no_cpu_op=raise python3 $@

