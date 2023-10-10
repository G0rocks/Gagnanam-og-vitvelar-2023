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

# Set random seed
random.seed(0)


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
    stops = numpy.zeros([n_stops, 5])

    # Put values in stops
    for i in range(0,n_stops):
        stops[i][0] = int(i+1)  # Number of stop on the way
        stops[i][1] = int(i*17+3)   # Stop ID
        stops[i][2] = (i-1)/(n_stops-1) # Percentage of stop (stop number/total number of stops on route)
        stops[i][3] = random.randint(0,2)   # Bulb space (how many buses can stop at this bus stop without interfering with other traffic)
        stops[i][4] = leid_type             # Leid type, 0 if it's a high speed & longer distance leid, 1 if it's a lower speed and shorter distane leid.

    return stops

def get_split_leidir(leidir: list, test_data_fraction: float):
    '''
    Takes in a dictionary with leidir and data attached to it and splits it according to the test_data_fraction

    Input:
    leidir: list of leidir
    test_data_fraction: float with percentage of how many of the leidir shall be used to test with. train_data_fraction = 1-test_data_fraction

    Output:
    train_leidir:   List of leidir to train with
    test_leidir:    List of leidir to test with
    '''
    # How many leidir
    n_leidir = len(leidir)

    # Split n_leidir for training and testing
    n_test_leidir = math.ceil(float(test_data_fraction)*n_leidir)

    # Go through all leidir
    i = 0
    train_leidir = []
    test_leidir = []
    for leid in leidir:
        if i < n_test_leidir:
            test_leidir.append(leid)
        else:
            train_leidir.append(leid)
        i = i+1

    print("Training leidir")
    print(train_leidir)
    print("Testing leidir")
    print(test_leidir)
    return train_leidir, test_leidir

def split_data(static_stop_data: dict, leidir_bus_stop_ratings: dict, test_data_fraction = 0.5):
    ''''
    Takes in a dictionary with bus stop data and a fraction (between 0 and 1, both included)
    test_data_fraction defaults to 0.5.

    Inputs:
    static_stop_data: dictionary with static bus stop data
    dynamic_stop_data: dictionary with static bus stop data
    test_data_fraction: float, defaults to 0.5. How much percentage (from 0 to 1) of the leidir in the data should be in the test features.

    Outputs:
    train_features: Numpy array with data
    test_features:  Numpy array with data
    train_targets:  Numpy array with data
    test_targets:   Numpy array with data
    '''
    print("Splitting data into training and test data sets")
    # Make sure test_data_fraction is good
    if test_data_fraction > 1:
        test_data_fraction = 1
    if test_data_fraction < 0:
        test_data_fraction = 0

    # Get list of leidir (keys)
    leidir = list(static_stop_data.keys())

    # Get list of which leidir will be used for training and which for testing
    train_leidir, test_leidir = get_split_leidir(leidir, test_data_fraction)

    # If fewer than 2 train_leidir or test_leidir, set the features and targets to contain the data from them
    if len(train_leidir) == 1:
        train_features = static_stop_data.get(train_leidir[0])
        train_targets = leidir_bus_stop_ratings.get(train_leidir[0])

    if len(test_leidir) == 1:
        test_features = static_stop_data.get(test_leidir[0])
        test_targets = leidir_bus_stop_ratings.get(test_leidir[0])

    # Loop through train leidir and concatenate the data
    print("Adding leid " + str(train_leidir[0]) + " to training features and targets")
    train_features = static_stop_data.get(train_leidir[0])
    train_targets = leidir_bus_stop_ratings.get(train_leidir[0])
    for i in range(1, len(train_leidir)):
        print("Adding leid " + str(train_leidir[i]) + " to training features and targets")
        train_features = numpy.concatenate((train_features, static_stop_data.get(train_leidir[i])), axis=0)
        train_targets = numpy.concatenate((train_targets, leidir_bus_stop_ratings.get(train_leidir[i])), axis=0)

    print("Adding leid " + str(test_leidir[0]) + " to testing features and targets")
    test_features = static_stop_data.get(test_leidir[0])
    test_targets = leidir_bus_stop_ratings.get(test_leidir[0])
    for i in range(1, len(test_leidir)):
        print("Adding leid " + str(test_leidir[i]) + " to testing features and targets")
        test_features = numpy.concatenate((test_features, static_stop_data.get(test_leidir[i])), axis=0)
        test_targets = numpy.concatenate((test_targets, leidir_bus_stop_ratings.get(test_leidir[i])), axis=0)

    # Return train and test features and targets
    print("Data split complete")    
    return train_features, test_features, train_targets, test_targets

def rate_1_bus_stop(n_stops: int, mean_time: float, std_dev_time: float, early_percent: float, leid_type: bool, n_stop: int) -> int:
    '''
    Rates a bus stop given the dynamic data of that specific stop.
    
    Inputs:
    n_stop:         Number of stop on the specific route
    mean_time:      Mean difference between actual departure time and planned departure time according to time table
    std_dev_time:   Standard deviation of the mean difference between actual and planned departure time
    early_percent:  Percentage of departures from this stop which were at least 1 minute earlier than the planned departure time
    leid_type:      0 for long distance & high speed, 1 for short distance & low speed

    Output:
    stop_rating:    Integer. Rating from 0-4. This is what the rating means:
        0: Awful, this bus stop needs immediate attention, analysis and action, it is not performing as it should
        1: Bad, this bus stop needs immediate attention and analysis. It is not performing as it should. Think about the action
        2: It could be worse but it could be better. This bus stop should get attention and analysis but the priority is lower than that of lower ratings.
        3: Good enough :) This bus stop is performing as it should, with the occasional mishap. It could do better but it's pointless to analyse bus stops with this rating unless you have already dealt with all other bus stops with lower ratings
        4: Excellent rating! It would take a miracel to have this bus stop performing any better than it is ^_^

    Notes:
        - We have higher expectations of leidir with leid_type 0 and lower expectations for leidir with leid_type 1
    '''
    # Default stop_rating at 3
    stop_rating = 3

    # If the bus leaves early a lot of the time (6 times out of every 300 times or more) then the stop_rating goes to
    if early_percent >= 0.02:
        stop_rating = 0
        if leid_type:
            stop_rating = stop_rating + 1
        
        return stop_rating
    if early_percent >= 0.01:
        stop_rating = 1
        if leid_type:
            stop_rating = stop_rating + 1
        
        return stop_rating
    if early_percent > 0:
        stop_rating = 2

    # High expectations of leidir with leid_type 0, lower expectations for leidir with leid_type 1
    if leid_type:
        stop_rating = stop_rating + 1

    # If the bus is late a lot, reduce stop_rating
    if mean_time > 5:
        stop_rating = stop_rating - 2
    elif mean_time > 3:
        stop_rating = stop_rating - 1
    
    # If the standard deviation is low, increase stop_rating
    if std_dev_time < 3:
        stop_rating = stop_rating + 1
    if std_dev_time < 2:
        stop_rating = stop_rating + 1
    if std_dev_time < 1:
        stop_rating = stop_rating + 1
    
    # As the number of stops grows, we become more relaxed when it comes to leaving precisely on time
    if n_stop > 15:
        stop_rating = stop_rating + 1
    
    if stop_rating < 0:
        stop_rating = 0
    if stop_rating > 4:
        stop_rating = 4

    # Return stop_rating
    return stop_rating

def rate_bus_stops_in_leid(leid_dynamic_stop_data: numpy.array) -> numpy.array:
    '''
    Rates all bus stops in leid, returns ratings

    Input:
    leid_dynamic_stop_data: Numpy.array with all the bus stops and data for this leid

    Output:
    stop_ratings: numpy.array with all the bus stop ratings
    '''
    # How many stops
    n_stops = leid_dynamic_stop_data.shape[0]

    # Create empty array without ratings
    stop_ratings = numpy.zeros([n_stops, 1])

    # Loop through each stop and rate it
    for i in range(n_stops):
        # Rate each stop
        n_stops         =   leid_dynamic_stop_data[i][0]
        mean_time       =   leid_dynamic_stop_data[i][1]
        std_dev_time    =   leid_dynamic_stop_data[i][2]
        early_percent   =   leid_dynamic_stop_data[i][3]
        leid_type       =   leid_dynamic_stop_data[i][4]
        n_stop          =   leid_dynamic_stop_data[i][5]

        stop_ratings[i] = rate_1_bus_stop(n_stops, mean_time, std_dev_time, early_percent, leid_type, n_stop)

    # Return stop_ratings
    return stop_ratings

def rate_bus_stops(dynamic_stop_data: dict) -> dict:
    '''
    Rates all the bus stops for all leidir in dynamic_stop_data. Returns the bus stops with ratings

    Input:
    bus_stops_dynamic_data: dictionary where the key is the name of the leid and the value is a 2D array with dynamic bus stop data

    Output:
    leidir_ratings: dictionary where the key is the name of the leid and the value is a 2D array with these values:
        stops[i]: Stop rating (value from 0 to 4 both included)
    '''
    # Initialize empty dictionary
    leidir_ratings = dict()

    # Get list of leidir (keys):
    leidir = list(dynamic_stop_data.keys())
    print("Rating bus stops with dynamic data in leidir: " + str(leidir))

    # For each leid, rate all bus stops
    for i in range(len(leidir)):
        # Add rating to dictionary
        print("Rating " + str(dynamic_stop_data.get(leidir[i]).shape[0]) + " bus stops for leid " + str(leidir[i]) + " with dynamic data")
        leidir_ratings[leidir[i]] = rate_bus_stops_in_leid(dynamic_stop_data.get(leidir[i]))

    # Return leidir_ratings
    return leidir_ratings
