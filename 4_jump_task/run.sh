#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run_model.py --task traj_loc_pred --model DeepMove --dataset jump --saved_model True