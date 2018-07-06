
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt,atan2
from pyproj import Proj, transform


def projections(x,y, inProj, outProj):
    """
    in this funtion we transform the data from the coordinate system to gps
    """
    longitd, latitud = transform(inProj,outProj,x,y)
    return latitud, longitd

def pythagoras(x, y):
    return np.sqrt(x**2 + y**2)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def pre_process_station_name(x):
    """
    Standarized the station names. This step is necesary to merge different data sets later
    """
    x = x.lower()        
    x = x.split()    
    return x[0]

def get_safe_random_value_normal(mean, std):
    """
    Get random values bigger than 0
    """

    while True:
        x = np.random.normal(mean, std, 1)[0]    
        if x > 0:
            break

    return x

