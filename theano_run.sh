nvidia-smi
THEANO_FLAGS=mode=FAST_RUN,device=gpu,init_gpu_device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES,print_active_device=True,enable_initial_driver_test=True,warn_float64=raise,force_device=True,assert_no_cpu_op=raise python3 $@
