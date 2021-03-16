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
from DefaultBuildings import Building_5ZoneAirCooled_SingleAgent, Building_5ZoneAirCooled
from CentralController import run_for_n_episodes
from Options import get_argparser
from SQLOutput import SQLOutput


def main(args):
    cobs.Model.set_energyplus_folder(global_paths["eplus"])

    parser = get_argparser()
    args   = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #
    # Define the building and the occupants
    if args.model == "5ZoneAirCooled_SingleAgent":
        building = Building_5ZoneAirCooled_SingleAgent()
    else:
        building = Building_5ZoneAirCooled()
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
            arguments_text += f"{arg:30} {arg_val}\n"
    f = open(os.path.join(args.checkpoint_dir, "options.txt"), "w")
    f.write(arguments_text)
    f.close()

    sqloutput = SQLOutput(os.path.join(args.checkpoint_dir, "ouputs.sqlite"), building)
    sqloutput.initialize()

    #
    # run the model for n episodes
    run_for_n_episodes(args.episodes_count, building, building_occ, args, sqloutput)

    sqloutput.db.commit()
    sqloutput.db.close()



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

