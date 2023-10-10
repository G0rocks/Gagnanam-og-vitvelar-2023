# Author:           Huldar
# Date:             2023-10-06
# Project:          Research project
# Acknowledgements: Strætó for hooking us up with the data from https://data02.straeto.is/data/raw/
'''
This program is a module containing helpful functions for dealing with the data from strætó
'''

# Packages
import random   # For randomizing things
import numpy    # To work with arrays
import math     # To roundup

def get_stops_from_leid(leid):
    '''
    Currently generating random stops
    inputs:
    <leid> Integer. Leiðin hjá strætó sem við ætlum að sækja stoppin fyrir

    output:
    <stops> Array 4xD. stops[0][0] er stop number, stops[0][1] er stop id, stops[0][2] er way_percent og stops[0][3] er bulb space
    '''
    # Check which type of leid we have. Mark with high speed & long distance as 0 vs low speed and shorter distance as 1.
    if leid == 1 or leid == 6:
        leid_type = 0
    else:
        leid_type = 1

    # Make stops between 20 and 45 stops per leid
    n_stops = random.randint(20,45)

    # Init empty stops list
    stops = numpy.zeros([n_stops, 4])

    # Put values in stops
    for i in range(0,n_stops):
        stops[i][0] = int(i+1)  # Number of stop on the way
        stops[i][1] = int(i*17+3)   # Stop ID
        stops[i][2] = (i-1)/(n_stops-1) # Percentage of stop (stop number/total number of stops on route)
        stops[i][3] = random.randint(0,2)   # Bulb space (how many buses can stop at this bus stop without interfering with other traffic)
        stops[i][4] = leid_type             # Leid type, 0 if it's a high speed & longer distance leid, 1 if it's a lower speed and shorter distane leid.

    return stops



def split_data(stop_data: dict, test_data_fraction=0.5):
    ''''
    Takes in a dictionary with bus stop data and a fraction (between 0 and 1, both included)
    test_data_fraction defaults to 0.5

    Output: train_set, test_set
    Both sets have the same 
    '''
    # How many leidir
    n_leidir = len(stop_data.keys())

    # Split n_leidir for training and testing
    n_test_leidir = math.ceil(test_data_fraction*n_leidir)
    n_train_leidir = n_leidir-n_test_leidir

    # Go through all stop data
    i = 0
    test_leidir = dict()
    train_leidir = dict()
    for key, value in stop_data.items():
        if i < n_test_leidir:
            test_leidir[key] = value
        else:
            train_leidir[key] = value
        i = i+1

    return train_leidir, test_leidir

        




