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
import random   # To randomly generate numbers

# Set random seed
random.seed(0)

def generate_bus_stop_static_data(leidir=[1,2,3,6]) -> dict:
    ''''
    Generates the static data about the bus stops.
    
    Input: List of leidir to look at.
    Output: Dictionary where the keys are the number of the route.
        Each value in the dictionary is a 2D numpy array where the first index (i) is the number of the stop for this route and the second index is by this list:
            stops[i][0]: Number of stop on the way
            stops[i][1]: Percentage of stop (stop number/total number of stops on route)
            stops[i][2]: Bulb space (how many buses can stop at this bus stop without interfering with other traffic)
            Stops[i][3]: Leid type, 0 if it's a high speed & longer distance leid, 1 if it's a lower speed and shorter distane leid.
    '''
    print("Generating static bus stop data")

    # Leiðir sem verið er að skoða (skilgreint í kalli falls)
    print("Leidir: " + str(leidir))

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
        stops[i][5]: number of this stop (equal to i if the array has not been randomized)
    '''
    print("Generating dynamic bus stop data")
    # Get list of leidir (keys):
    leidir = list(static_stop_data.keys())
    print("Leidir: " + str(leidir))

    # Initialize empty dynamic_stop_data dictionary
    dynamic_stop_data = dict()

    # Loop through all leidir and generate bus stop data for each bus stop on each leid
    for i in range(len(leidir)):
        # Get static data from leid
        n_stops_on_leid = static_stop_data.get(leidir[i]).shape[0]

        # Loop through all stops and generate the dynamic data for that stop
        stops = numpy.zeros([n_stops_on_leid, 6])
        for j in range(n_stops_on_leid):
            mean_departure_diff = random.randint(1,75)/10   # Mean departure difference
            std_dev_departure = random.randint(5,75)/10     # Standard deviation of departure difference
            early_percent = random.randint(0,3)/100         # Percentage of early departures
            leid_type = static_stop_data.get(leidir[i])[j][4]   # Type of leid

            stops[j][0] = n_stops_on_leid # Number of stops on the way
            stops[j][1] = mean_departure_diff# Mean difference between actual and planned departure time
            stops[j][2] = std_dev_departure # Standard deviation of the mean difference between actual and planned departure time
            stops[j][3] = early_percent # Percentage of departures from this stop which were at least 1 minute earlier than the planned departure time
            stops[j][4] = leid_type # leid_type. 0 for long distance & high speed, 1 for short distance & low speed
            stops[j][5] = j+1         # Number of this stop

        # Add stops array to leid in dynamic_data
        dynamic_stop_data[leidir[i]] = stops

    # Return dynamic_stop_data
    return dynamic_stop_data