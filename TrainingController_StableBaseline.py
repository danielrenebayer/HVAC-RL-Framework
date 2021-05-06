
import os
import sys
import pickle
import sqlite3
import datetime

from global_paths import global_paths
if not global_paths["COBS"] in sys.path: sys.path.append( global_paths["COBS"] )
import cobs

from BuildingOccupancy import BuildingOccupancyAsMatrix
import DefaultBuildings
from Options import get_argparser
from SQLOutput import SQLOutput

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from StableBaselineEnv import StableBaselineEnv


def main(args = None):
    cobs.Model.set_energyplus_folder(global_paths["eplus"])

    parser = get_argparser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    if args.continue_training and not os.path.exists(os.path.join(args.checkpoint_dir, "status.pickle")):
        args.continue_training = False
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    episode_offset = 0
    status_dict    = {}

    #
    # Define the building and the occupants
    if args.model not in DefaultBuildings.__dict__.keys():
        raise AttributeError(f"{args.model} is no model in DefaultBuildings.py!")
    building = DefaultBuildings.__dict__[ args.model ](args)

    #sqloutput = SQLOutput(os.path.join(args.checkpoint_dir, "ouputs.sqlite"), building)
    #if not args.continue_training:
    #    sqloutput.initialize()
    #else:
    #    f = open(os.path.join(args.checkpoint_dir, "status.pickle"), "rb")
    #    status_dict = pickle.load(f)
    #    f.close()

    if args.continue_training:
        # load the building_occ object
        f = open(os.path.join(args.checkpoint_dir, "building_occ.pickle"), "rb")
        building_occ = pickle.load(f)
        f.close()
        # set episode_offset
        episode_offset = status_dict["next_episode_offset"]
        # define the latest model paths
        args.load_models_from_path = args.checkpoint_dir
        args.load_models_episode   = episode_offset - 1
    else:
        # initialize a new building object
        building_occ = BuildingOccupancyAsMatrix(args, building)
        #
        # save the building_occ object
        f = open(os.path.join(args.checkpoint_dir, "building_occ.pickle"), "wb")
        pickle.dump(building_occ, f)
        f.close()

    #
    # save the arguments as text
    arguments_text = ""
    for arg, arg_val in vars(args).items():
        arg_def = parser.get_default(arg)
        if arg_def != arg_val:
            arguments_text += f"{arg:30} {arg_val:20} [Default: {arg_def}]\n"
        else:
            arguments_text += f"{arg:30} {arg_val:20}\n"
    options_filename = "options.txt" if not args.continue_training else f"options_episode_offset_{episode_offset}.txt"
    f = open(os.path.join(args.checkpoint_dir, options_filename), "w")
    f.write(arguments_text)
    f.close()

    #
    # run the model for n episodes
    env = StableBaselineEnv(building, building_occ, args)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.episodes_count*args.ts_per_hour*24*30, log_interval=30)
    model.save(os.path.join(args.checkpoint_dir, "dqn_model.stablebaseline"))
    #check_env(env)
    while not env.building.model_is_terminate():
        env.step(0)
    #
    #

    #sqloutput.db.commit()
    #sqloutput.db.close()

    # write the status object
    status_dict["next_episode_offset"] = args.episodes_count + episode_offset
    f = open(os.path.join(args.checkpoint_dir, "status.pickle"), "wb")
    pickle.dump(status_dict, f)
    f.close()

    print(" -- Finished -- ")



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

