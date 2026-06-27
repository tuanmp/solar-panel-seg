#!/bin/bash
# gpu_requirement="cpu"
gpu_requirement="gpu&hbm80g"
salloc -A m2616_g -C $gpu_requirement -q shared_interactive --nodes 1 --ntasks-per-node 1 --gpus-per-task 1 --cpus-per-task 32 --time 01:00:00 --signal=SIGUSR1@180 #--image=tuanpham1503/torch_conda:0.4
