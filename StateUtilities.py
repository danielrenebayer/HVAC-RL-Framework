
import datetime
from copy import deepcopy
import torch


variable_ranges = {
    "Day of Week":             (  0.0,    6.0),
    "Minutes of Day":          (  0.0, 1439.0), # one day has 24*60-1 minutes
    "Calendar Week":           (  1.0,   53.0),
    "Outdoor Air Temperature": (-20.0,   40.0), 
    "Outdoor Air Humidity":    (  0.0,  100.0),
    "Outdoor Wind Speed":      (  0.0,   15.0), # TODO: search for true maximal value
    'Outdoor Wind Direction':  (  0.0,  360.0),
    'Outdoor Solar Radi Diffuse': (0.0,1000.0),
    'Outdoor Solar Radi Direct':  (0.0, 200.0),
    "Zone VAV Reheat Damper Position": (0.0,1.0),
    "Zone Relative Humidity":  (  0.0,  100.0),
    "Zone CO2":                (410.0, 5000.0),
    "Zone Temperature":        ( 10.0,   40.0),
    "Zone Heating/Cooling-Mean Setpoint": ( 14.0,30.0),
    "Zone Heating/Cooling-Delta Setpoint":(  0.1,10.0)
}


def normalize_variable(v, varname):
    vmin, vmax = variable_ranges[varname]
    v = 2 * (v - vmin) / (vmax - vmin) - 1
    if   v <-1: return-1.0
    elif v > 1: return 1.0
    return v


def backtransform_variable(vi, varname):
    if   vi < -1: vi = -1.0
    elif vi >  1: vi =  1.0
    vmin, vmax = variable_ranges[varname]
    return (vi + 1) * (vmax - vmin) / 2 + vmin


def normalize_variables_in_dict(vardict, inplace=False):
    if inplace:
        outdict = vardict
    else:
        outdict = deepcopy(vardict)

    for k in outdict.keys():
        for known_variable in variable_ranges.keys():
            if k.endswith(known_variable):
                outdict[k] = normalize_variable( outdict[k], known_variable )
                break
    return outdict


def backtransform_variables_in_dict(vardict, inplace=False):
    if inplace:
        outdict = vardict
    else:
        outdict = deepcopy(vardict)

    for k in outdict.keys():
        for known_variable in variable_ranges.keys():
            if k.endswith(known_variable):
                outdict[k] = backtransform_variable( outdict[k], known_variable )
                break
    return outdict


def expand_state_timeinfo_temp(state, building):
    """
    Expands the state dictionary by Minutes of Day, Day of Week, Calendar Week and splits the zone temperatures.
    """
    # expand time information
    state["Minutes of Day"] = state['time'].hour*60 + state['time'].minute
    state["Day of Week"]    = state['time'].weekday()
    state["Calendar Week"]  = state['time'].isocalendar()[1]
    # isocalendar: Return a 3-tuple containing ISO year, week number, and weekday
    #
    # expand temperature for all zones
    for r in building.room_names:
        state[f"{r} Zone Temperature"] = state["temperature"][r]


def normalized_state_to_tensor(state, building):
    """
    Transforms a state dict to a pytorch tensor.
    The function ensures the correct ordering of the elements according to the list building.global_state_variables.

    It expects a **normalized** state as input.
    """
    ten = [[ state[sval] for sval in building.global_state_variables ]]
    return torch.tensor(ten)


def unnormalized_state_to_tensor(state, building):
    """
    Transforms a state dict to a pytorch tensor.
    The function ensures the correct ordering of the elements according to the list building.global_state_variables.

    It expects a **unnormalized** state as input.
    """
    expand_state_timeinfo_temp(state, building)
    return normalized_state_to_tensor(normalize_variables_in_dict(state), building)


def fix_year_confussion(state):
    """
    Because of an error in the EPlus-Python-interface (it passes the wrong year),
    we have to manually set the year to 2017 (which is the default EPlus year).
    """
    orig_date_obj = state["time"]
    new_date_obj  = datetime.datetime(
                        year  = 2017,
                        month = orig_date_obj.month,
                        day   = orig_date_obj.day,
                        hour  = orig_date_obj.hour,
                        minute= orig_date_obj.minute)
    state["time"] = new_date_obj


def retrieve_substate_for_agent(normalized_state, agent, building):
    """
    Returns a subset of the state dictionary only containing the entries that are required by the agent.
    It will also change the keys to fit to the expected key names of the agent.
    
    The state should already be normalized in order to pass it to the agent, because the names of the keys may have changed.
    """
    state_subset = {}
    for k in agent.input_parameters:
        if k in normalized_state.keys():
            state_subset[k] = normalized_state[k]
        else:
            z_prefix, _ = building.agent_device_pairing[agent.name]
            state_subset[k] = normalized_state[f"{z_prefix} {k}"]
    return state_subset


