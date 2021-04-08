#
# This training script initializes a complete new simulation run
# and trains all agents
#
# Building              5ZoneAirCooled
# Wheather              Chicage O'Hare Airport
# Episode period        1. July to 30. July
# Number of occupants   40
# Time resolution       5 minutes
# Number of training episodes   100
# Critic hidden layer size      40
#

import os
import sys
import pickle
import sqlite3
import datetime

from global_paths import global_paths
if not global_paths["COBS"] in sys.path: sys.path.append( global_paths["COBS"] )
import cobs

from BuildingOccupancy import BuildingOccupancy
import DefaultBuildings
from CentralController import run_for_n_episodes, one_baseline_episode
from Options import get_argparser
from SQLOutput import SQLOutput


def main(args):
    cobs.Model.set_energyplus_folder(global_paths["eplus"])

    parser = get_argparser()
    args   = parser.parse_args()
    if args.continue_training and not os.path.exists(os.path.join(args.checkpoint_dir, "status.pickle")):
        args.continue_training = False
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    episode_offset = 0
    status_dict    = {}

    #
    # Define the building and the occupants
    if args.model == "5ZoneAirCooled_SingleAgent":
        building = DefaultBuildings.Building_5ZoneAirCooled_SingleAgent(args)
    elif args.model == "5ZoneAirCooled_SmallAgents":
        building = DefaultBuildings.Building_5ZoneAirCooled_SmallAgents(args)
    elif args.model == "5ZoneAirCooled_SmallSingleAgent":
        building = DefaultBuildings.Building_5ZoneAirCooled_SmallSingleAgent(args)
    else:
        building = DefaultBuildings.Building_5ZoneAirCooled(args)

    sqloutput = SQLOutput(os.path.join(args.checkpoint_dir, "ouputs.sqlite"), building)
    if not args.continue_training:
        sqloutput.initialize()
    else:
        f = open(os.path.join(args.checkpoint_dir, "status.pickle"), "rb")
        status_dict = pickle.load(f)
        f.close()

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
        building_occ = BuildingOccupancy()
        building_occ.set_room_settings(building.room_names[:-1], {building.room_names[-1]: 40}, 40)
        building_occ.generate_random_occupants(args.number_occupants)
        building_occ.generate_random_meetings(15,0)
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
    # call the controlling function
    if args.algorithm == "rule-based":
        # run one sample episode using the rule-based agent
        outputs = one_baseline_episode(building, building_occ, args, sqloutput)
        f = open(os.path.join(args.checkpoint_dir, "complete_outputs.pickle"), "wb")
        pickle.dump(outputs, f)
        f.close()
    else:
        # run the model for n episodes
        run_for_n_episodes(args.episodes_count, building, building_occ, args, sqloutput, episode_offset)

    sqloutput.db.commit()
    sqloutput.db.close()

    # write the status object
    status_dict["next_episode_offset"] = args.episodes_count + episode_offset
    f = open(os.path.join(args.checkpoint_dir, "status.pickle"), "wb")
    pickle.dump(f, status_dict)
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
        main(sys.argv)

