# Author:           Huldar
# Date:             2023-10-06
# Project:          Research project
# Acknowledgements: Strætó for hooking us up with the data from https://data02.straeto.is/data/raw/
'''
This program gets unlabeled data from Strætó about bus stops and bus stop timetables as well as data about bus trips
with timestamps where you can tell when the bus left the bus stop.
'''

# Modules used
import pandas   # Used for reading data from raw data files
import numpy    # Used to work with arrays
import custom_straeto_tools # Includes helpful functions for dealing with the data from strætó

def generate_bus_stop_static_data(leidir=[1,2,3,6]) -> dict:
    ''''
    Generates the static data about the bus stops.
    
    Input: List of leidir to look at.
    Output: Dictionary where the keys are the number of the route.
        Each value in the dictionary is a 2D numpy array where the first index (i) is the number of the stop for this route and the second index is by this list:
            stops[i][0]: Number of stop on the way
            stops[i][1]: Stop ID
            stops[i][2]: Percentage of stop (stop number/total number of stops on route)
            stops[i][3]: Bulb space (how many buses can stop at this bus stop without interfering with other traffic)
            Stops[i][4]: Leid type, 0 if it's a high speed & longer distance leid, 1 if it's a lower speed and shorter distane leid.
    '''
    print("Generating labeled data using data from strætó")

    # Leiðir sem verið er að skoða (skilgreint í kalli falls)
    print("Skoðum leiðir: " + str(leidir))

    # Gera stoppistodvar_leida sem inniheldur stoppistodvar fyrir leidir
    stoppistodvar_leida = dict()
    for leid in leidir:
        # Add stoppistodvar leida to dictionary
        stoppistodvar_leida[leid] = custom_straeto_tools.get_stops_from_leid(leid)

    return stoppistodvar_leida

def generate_bus_stop_dynamic_data(static_stop_data: dict) -> dict:
    ''''
    Generates the dynamic data about the bus stops.

    Input: static_stop_data dictionary where the keys are the number of the route
            and the values are 2D numpy array with N rows and 4 columns.
    Output: Dictionary where the keys are the number of the route.
        Each value in the dictionary is a 2D numpy array where the first index (i) is the number of the stop for this route and the second index is by this list:
        stops[i][0]: Number of stop on the way
        stops[i][1]: Mean difference between actual departure time and planned departure time according to time table
        stops[i][2]: Standard deviation of the mean difference between actual and planned departure time
        stops[i][3]: Percentage of departures from this stop which were at least 1 minute earlier than the planned departure time
        stops[i][4]: Leid_type. 0 for long distance & high speed, 1 for short distance & low speed
    '''

    pass

def rate_1_bus_stop(n_stop: int, mean_time: float, std_dev_time: float, early_percent: float, leid_type: bool) -> int:
    '''
    Rates a bus stop given the dynamic data of that specific stop.
    
    Inputs:
    n_stop:         Number of stop on the specific route
    mean_time:      Mean difference between actual departure time and planned departure time according to time table
    std_dev_time:   Standard deviation of the mean difference between actual and planned departure time
    early_percent:  Percentage of departures from this stop which were at least 1 minute earlier than the planned departure time
    Leid_type:      0 for long distance & high speed, 1 for short distance & low speed

    Output:
    stop_rating:    Integer. Rating from 0-4. This is what the rating means:
        0: Awful, this bus stop needs immediate attention, analysis and action, it is not performing as it should
        1: Bad, this bus stop needs immediate attention and analysis. It is not performing as it should. Think about the action
        2: It could be worse but it could be better. This bus stop should get attention and analysis but the priority is lower than that of lower ratings.
        3: Good enough :) This bus stop is performing as it should, with the occasional mishap. It could do better but it's pointless to analyse bus stops with this rating unless you have already dealt with all other bus stops with lower ratings
        4: Excellent rating! It would take a miracel to have this bus stop performing any better than it is ^_^
    '''
    pass

def rate_bus_stops(dynamic_stop_data: dict) -> dict:
    '''
    Rates all the bus stops for all leidir in dynamic_stop_data. Returns the rated 
    '''

    # Use rate_1_bus_stop!!!!
    pass