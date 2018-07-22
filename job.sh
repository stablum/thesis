#!/bin/bash --login
#$ -l h_rt=168:00:00

#### ignored option -l gpu=GTX480
#### other possibility: -l gpu=GTX680
#### other possibility: -l fat,gpu=K20
#### other possibility: -l gpu=C2050
#### other possibility: -l gpu=GTX480

module add python/3.5.2
source ~/venv5/bin/activate
cd thesis
nvidia-smi
echo -n "hostname:"
hostname
#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fstablum/.local/lib
EXTENSION=$(echo $1 | sed 's/.*\.//g')
echo EXTENSION:$EXTENSION
case "$EXTENSION" in
    py)
        INTERPRETER=python3
        ;;
    sh)
        INTERPRETER=bash
        ;;
    *)
        echo $1 does not have an identifiable extension, @=$@
        exit
esac
echo INTERPRETER:$INTERPRETER
OMP_NUM_THREADS=8 THEANO_FLAGS=mode=FAST_RUN,device=cuda,init_gpu_device=cuda,floatX=float32,nvcc.flags=-D_FORCE_INLINES,print_active_device=True,enable_initial_driver_test=True,warn_float64=raise,force_device=True,assert_no_cpu_op=raise,allow_gc=False $INTERPRETER $@
#THEANO_FLAGS=floatX=float32,warn_float64=raise $INTERPRETER $@

