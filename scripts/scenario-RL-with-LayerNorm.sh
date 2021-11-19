#!/bin/bash

#
# This is an example file for starting the training
#
# It restarts the simulation every 160 iterations.
# Comment this out if your system works stable and does not end long processes.
#
# This example uses no pretraining, but shared parameters.
# After 160 episodes one evaluation episode (no training, only evaluation) is included.
#

set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/RL-with-LayerNorm")"/${datestr}"
num_iters=100
num_episodes_per_iter=160

let epsilon_final=$num_iters*$num_episodes_per_iter
let num_iters_half=$num_iters/2
let num_iters_quart=$num_iters/4
let num_iters_threequart=3*$num_iters/4
mkdir -p $checkpoint_dir

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--ddqn_new" ) # Double DDQN
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
    #
    # use shared parameters until the end of training
    #if (( $i < $num_iters_threequart )); then
        arguments+=( "--shared_network_per_agent_class" )
    #fi
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    #
    # Learning rate decay
    if   (( $i < $num_iters_quart )); then
        arguments+=( "--lr" 0.1 )
    elif (( $i < $num_iters_half )); then
        arguments+=( "--lr" 0.065 )
    elif (( $i < $num_iters_threequart )); then
        arguments+=( "--lr" 0.04 )
    else
        arguments+=( "--lr" 0.02 )
    fi
    #
    arguments+=( "--discount_factor" 0.9 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--reward_offset" 0.3 )
    arguments+=( "--lambda_rwd_energy" 0.017 )
    arguments+=( "--lambda_rwd_mstpc"  0.16 )
    arguments+=( "--clip_econs_at" 150.0 )
    arguments+=( "--energy_cons_in_kWh" )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 2 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" $epsilon_final )
    arguments+=( "--epsilon_decay_mode" "linear" )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--use_layer_normalization" )
    arguments+=( "--checkpoint_dir" $checkpoint_dir )
    arguments+=( "--idf_file" $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) )
    arguments+=( "--epw_file" "../../COBS/cobs/data/weathers/8.epw" )
    arguments+=( "--episode_start_month" 1 )
    arguments+=( "--continue_training" )

    python ../code/TrainingController.py "${arguments[@]}" | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

