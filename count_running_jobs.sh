#!/bin/bash

module load slurm

squeue | grep fstablum | count -l
