# HVAC RL Framework

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#)[![License](http://img.shields.io/badge/license-MIT-green.svg?style=flat)](https://github.com/sustainable-computing/COBS/blob/master/LICENSE)

This framework contains the code of our publication for IEEE SusTech 2022.
It is developed by Daniel Bayer.
A special thanks goes to Niklas Ebell and Oliver Birkholz for the fruitful discussions.

This framework is licensed under the MIT license.

## Dependencies
Our framework requires [EnergyPlus](https://github.com/NREL/EnergyPlus/releases/tag/v9.3.0) in Version 9.3.
Moreover it requires [COBS](https://github.com/sustainable-computing/COBS)

Further requirements can be found in the requirements.txt file.

## Acknowledgement
Building models and weather data (`./scripts/data/*`): The Pacific Northwest National Laboratory (PNNL)

## Installation
It is recommended to place COBS and EnergyPlus above the current folder.
Anyway, the paths in the file `code/global_paths.py` should be ajusted to your individual settings.
Detailed example scripts can be found in the `scripts` subfolder.

