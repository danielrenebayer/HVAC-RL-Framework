#!/bin/bash

#
# Baseline 010
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b010")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "baseline_rule-based" \
	--ts_until_regulation 0 \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--stp_reward_step_offset 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--episode_start_month 1


