"""
Data load and transform utilities

"""

import pandas as pd
import random
import eviltransform
from time import time
import logging
import numpy as np
import math

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


# DIDI data schema
NAMES = ['driverid', 'orderid', 'timestamp', 'lon', 'lat']

# Study area
pts_WGS = [(34.241, 108.943), (34.241, 108.9415), (34.234, 108.9415), (34.234, 108.943)]
pts_GCJ = [eviltransform.wgs2gcj(p, q) for (p, q) in pts_WGS]
pts = pd.DataFrame(pts_GCJ, columns=['lat', 'lon'])

lat_min = pts.lat.min()
lat_max = pts.lat.max()
lon_min = pts.lon.min()
lon_max = pts.lon.max()

# Mid-point of the East-West study area
ref_lon = (lon_min + lon_max)/2.0

# Parameter (in meters) to determine trip direction
THRESHOLD = 200


def fetch(filename, bbox, threshold):
    """
    Loads a single day's worth of data as a dataframe within a
    specified bounding box.
    bbox = [lat_min, lat_max, lon_min, lon_max]
    threshold = meters from bounding box edge to consider in trip direction
    """

    start_time = time()
    reader = pd.read_csv(filename,
                         compression='gzip',
                         iterator=True,
                         chunksize=10**6,
                         header=None, names=NAMES)

    lat_min, lat_max, lon_min, lon_max = bbox

    # Mid-point of the East-West study area
    ref_lon = (lon_min + lon_max)/2.0

    df = pd.concat([p[(p.lat >= lat_min) & (p.lat <= lat_max) &
                      (p.lon >= lon_min) & (p.lon <= lon_max)] for p in reader])

    logger.info("File: {}".format(filename))
    logger.info("Read data ({} rows) in {:2f} sec".format(len(df), time() - start_time))
    logger.info("Unique trips: {}, unique drivers: {}".format(len(df.orderid.unique()),
                                                              len(df.driverid.unique())))

    # Get trip direction
    trip_dir = (df
                .groupby('orderid')
                .apply(lambda x: trip_direction(x, lat_max, ref_lon, lat_min, ref_lon, threshold)))

    trip_dir.rename("direction", inplace=True)
    df = df.join(trip_dir, on='orderid', how='left')

    # manage time zones (Chinese time zones is GMT +8)
    df.timestamp = pd.to_datetime(df.timestamp, unit='s')
    df.timestamp += pd.Timedelta('8 hours')

    # Linearize space
    df['distance'] = df.apply(linearize, args=(lat_max, ref_lon, lat_min, ref_lon), axis=1)

    return df


def trip_state(trip):
    """ Trip/orderid level state """

    trip = trip.reset_index()

    # distance traveled within time window (meters)
    d = trip.distance.max() - trip.distance.min()

    # time it took this trip within time window (seconds)
    t = pd.Timedelta(trip.timestamp.max() - trip.timestamp.min()).total_seconds()

    return pd.Series([d, t],
                     index=['distance', 'traveltime'])


def space_mean_speed(time_grp):
    """ Link level state """

    trips = time_grp.reset_index().groupby('orderid')
    state = trips.apply(trip_state)

    # number of trips in this window/direction
    num_trips = len(trips)

    # Link level measures for this time window
    d = state.distance.sum()
    t = state.traveltime.sum()

    # Correct space mean speed (km/h)
    with np.errstate(all='raise'):
        try:
            s = (d * 3.6) / t
        except FloatingPointError:
            s = np.nan

    return pd.Series([d, t, s, num_trips],
                     index=['distance', 'traveltime', 'speed', 'numtrips'])


def generate_time_series(df):
    """
    Generates the link level time series with
    space mean speeds per 5 minute intervals
    Takes as input a DataFrame that is returned from `fetch` (above) """

    start_time = time()
    logger.info("Generating time series ...")
    c1 = df.direction.isin(['north', 'south'])
    time_groups = df[c1].set_index('timestamp').groupby([pd.Grouper(freq='5Min'), 'direction'])

    sms = time_groups.apply(space_mean_speed)

    logger.info("Done in {:2f} sec".format(time() - start_time))

    return sms.reset_index()


def raw_state_variables(df):
    """ Get raw travel times/speeds """
    c1 = df.direction.isin(['north', 'south'])
    df_state = df[c1].groupby('orderid').apply(__state_variables)
    return df_state


def __state_variables(trip):
    """ Derive the observed state variables for a trip """

    # distance traveled within segment (meters)
    d = trip.distance.max() - trip.distance.min()

    # time it took this trip (seconds)
    t = pd.Timedelta(trip.timestamp.max() - trip.timestamp.min()).total_seconds()

    # speed (km/h)
    s = (d * 3.6)/t

    return pd.DataFrame({'start': trip.timestamp.min(),
                         'direction': trip.direction.unique(),
                         'distance': d,
                         'traveltime': t,
                         'speed': s})


def linearize(x, lat_max, lon_max, lat_min, lon_min):
    """
    Since the roads are vertical, we can directly linearize
    w.r.t. to an appropriate anchor point which depends on the
    direction.
    """
    if x.direction == 'north':
        # Measure progress from the southern edge
        return distance(x.lat, x.lon, lat_min, lon_min)

    elif x.direction == 'south':
        # Measure it from the Northern edge
        return distance(x.lat, x.lon, lat_max, lon_max)

    else:
        # Invalid trajectories/None
        return np.nan


def distance(lat1, lon1, lat2, lon2):
    """ distance in meters """

    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d * 1000


def trip_direction(x, top_lat, top_lon, bot_lat, bot_lon, THRESHOLD):
    """ A simple heuristic that assigns a directionality to a trip x """

    # find earliest position update
    earliest = x.loc[x.timestamp.idxmin()]

    # find latest position update
    latest = x.loc[x.timestamp.idxmax()]

    # Measure how far these are from the study boundaries
    early_top = distance(earliest.lat, earliest.lon, top_lat, top_lon)
    early_bot = distance(earliest.lat, earliest.lon, bot_lat, bot_lon)
    later_top = distance(latest.lat, latest.lon, top_lat, top_lon)
    later_bot = distance(latest.lat, latest.lon, bot_lat, bot_lon)

    if early_bot <= THRESHOLD:
        if later_top <= THRESHOLD:
            return "north"
        else:
            return "invalid"

    if early_top <= THRESHOLD:
        if later_bot <= THRESHOLD:
            return "south"
        else:
            return "invalid"
