#!/bin/bash

module load slurm
squeue -o '%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R %f %L' 
