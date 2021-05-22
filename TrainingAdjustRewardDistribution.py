#
# Use this program to find a reward scale so that
# reward distributions are adjusted.
#

import os
import sys
import pickle
import sqlite3
import datetime

import numpy as np
import pandas as pd
import scipy.special
import scipy.spatial.distance

from global_paths import global_paths
if not global_paths["COBS"] in sys.path: sys.path.append( global_paths["COBS"] )
import cobs

from BuildingOccupancy import BuildingOccupancyAsMatrix
import DefaultBuildings
import Agents
from CentralController import run_for_n_episodes
from Options import get_argparser
from SQLOutput import SQLOutput

#
# Function taken from notebooks/visualization_helper_v2.py
#
def convert_sqlite_to_df(db_conn):
    tables = ["eels", "sees", "seesea", "sees_er"]
    dfs = {tname: None for tname in tables}
    for table in tables:
        dfs[table] = pd.read_sql(f"SELECT * from {table};", db_conn)
        print(f"Table {table} convertet to a pandas dataframe.")
    return dfs


def main(args = None):
    cobs.Model.set_energyplus_folder(global_paths["eplus"])

    parser = get_argparser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    if not os.path.exists(os.path.join(args.checkpoint_dir, "status.pickle")):
        raise RuntimeError(f"status.pickle not found in {args.checkpoint_dir}")
    eval_dir = os.path.join(args.checkpoint_dir, "RewardDistributionEval")
    os.makedirs(eval_dir, exist_ok=True)
    episode_offset = 0
    status_dict    = {}

    #
    # Define the building and the occupants
    if args.model not in DefaultBuildings.__dict__.keys():
        raise AttributeError(f"{args.model} is no model in DefaultBuildings.py!")
    building = DefaultBuildings.__dict__[ args.model ](args)

    sqloutput = SQLOutput(os.path.join(eval_dir, "ouputs-0.sqlite"), building)
    sqloutput.initialize()

    f = open(os.path.join(args.checkpoint_dir, "status.pickle"), "rb")
    status_dict = pickle.load(f)
    f.close()
    # load the building_occ object
    f = open(os.path.join(args.checkpoint_dir, "building_occ.pickle"), "rb")
    building_occ = pickle.load(f)
    f.close()
    # set episode_offset
    episode_offset = status_dict["next_episode_offset"]
    # define the latest model paths
    args.load_models_from_path = args.checkpoint_dir
    args.load_models_episode   = episode_offset - 1

    #
    # Run evaluation episode for rulebased reward function
    args.reward_function = "rulebased_agent_output"
    args.network_storage_frequency = episode_offset + 20
    print(f"args.network_storage_frequency = {args.network_storage_frequency}")
    print("Run evaluation episode using the latest saved networks.")
    run_for_n_episodes(1, building, building_occ, args, sqloutput, episode_offset, True)
    sqloutput.db.commit()

    print("Compute target distribution")
    dfs = convert_sqlite_to_df(sqloutput.db)
    target_reward_hist, hist_labels = np.histogram(dfs["sees"].loc[:, "reward"], bins=20, range=(-8,0.1), density=True)
    sqloutput.db.close()

    #
    # Loop until we found a good parameter setting
    n_run = 1
    best_jsd_value  = []
    best_jsd_rscale = []
    best_jsd_hist   = []
    #for reward_scale in np.arange(0.05, 1.75, 0.05):
    #for reward_scale in np.arange(0.005, 0.1, 0.004):
    # run one episode with reward scale 1 and then scale again in a loop
    for _ in range(1):
        sqloutput = SQLOutput(os.path.join(eval_dir, f"ouputs-{n_run}.sqlite"), building)
        sqloutput.initialize()
        #
        Agents.QNetwork._agent_class_shared_networks = {}
        Agents.QNetwork._agent_class_shared_n_count  = {}
        #
        # Run the simulation
        args.reward_function = "sum_energy_mstpc"
        args.reward_scale    = 1
        run_for_n_episodes(1, building, building_occ, args, sqloutput, episode_offset, True)
        sqloutput.db.commit()
        # Compute current distribution
        dfs = convert_sqlite_to_df(sqloutput.db)
        for reward_scale in np.arange(0.01, 2.95, 0.01):
            current_reward_hist, _ = np.histogram(dfs["sees"].loc[:, "reward"]*reward_scale, bins=20, range=(-8,0.1), density=True)
            jsd = scipy.spatial.distance.jensenshannon(target_reward_hist, current_reward_hist)
            #
            best_jsd_value.append(jsd)
            best_jsd_rscale.append(reward_scale)
            best_jsd_hist.append(current_reward_hist)
            print(f"reward scale = {reward_scale:6.5f}, JSD = {jsd:11.8f}")
        #
        sqloutput.db.close()
        n_run += 1
    # find optimal value
    position = np.argmin(best_jsd_value)
    print()
    print("All results:")
    for jsd, rscale in zip(best_jsd_value, best_jsd_rscale):
        print(f"rscale = {rscale:4.2f}, JSD = {jsd:11.8f}")
    print()
    print(f"Optimal jsd value = {best_jsd_value[position]:11.8f} for reward scale = {best_jsd_rscale[position]:6.5f}")
    # save things to a pickle file
    optimal_settings = {
        "target_reward_hist":target_reward_hist,
        "best_fitting_reward_hist":best_jsd_hist[position],
        "best_jds":best_jsd_value[position],
        "best_reward_scale":best_jsd_rscale[position],
        "labels":hist_labels,
        "all_jsds":best_jsd_value,
        "all_jsds_rscales":best_jsd_rscale,
        "reward_hist_for_scale_1":np.histogram(dfs["sees"].loc[:, "reward"], bins=20, range=(-8,0.1), density=True),
        "all_rewards":best_jsd_hist
    }
    f = open(os.path.join(eval_dir, "best-results.pickle"), "wb")
    pickle.dump(optimal_settings, f)
    f.close()
    f = open(os.path.join(eval_dir, "best-reward-scale.txt"), "w")
    f.write(str(best_jsd_rscale[position]))
    f.close()
    f = open(os.path.join(eval_dir, "name-of-used-model.txt"), "w")
    load_path = os.path.abspath(args.load_models_from_path)
    f.write(f"{load_path}/episode_{args.load_models_episode}_agent_0")
    f.close()




if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == '--configfile':
        cfile = open(sys.argv[2], 'r')
        args_raw = cfile.readlines()
        cfile.close()
        args_parsed = []
        for ln in args_raw:
            args_parsed.extend( ln.split() )
        main(args_parsed)
    else:
        main()

