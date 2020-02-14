# -*- coding: utf-8 -*-
# Install dependencies
import requests
import datetime
import pandas as pd
import numpy as np
from IPython import display # For updating the cell dynamically
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

# Set fontsize
plt.rcParams.update({'font.size': 24})

# Create a function for requesting the data
"""
Adapted from from Stefanie Molin: Hands-On-Data-Analysis-With-Pandas. Available at:
https://github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas/tree/master/ch_04
"""


def make_request(endpoint, payload=None):
    """
    Make a request to a specific endpoint on the weather API
    passing headers and optional payload.

    Parameters:
        - endpoint: The endpoint of the API you want to
                    make a GET request to.
        - payload: A dictionary of data to pass along
                   with the request.

    Returns:
        Response object.
    """
    return requests.get(
        f'https://www.ncdc.noaa.gov/cdo-web/api/v2/{endpoint}',
        headers={
            'token': 'INSERT TOKEN HERE'
        },
        params=payload
    )


def station_lookup(fips_id):
    """
    This function grabs the station metadata for the given region entered by the user.
    It can be helpful when identifying which station to use and what the stationid is.

    Parameters:
        - fipsid: A string containing the FIPS ID (ex. UK)

    Returns:
        - df: A pandas dataframe containing the elevation, start date, end date, latitude, longitude,
              name of the station, data coverage (0 to 1), stationid, and elevation units
    """

    station_id = make_request('stations',
                              {
                                  'datasetid': 'GHCND',  # Global Historical Climatology Network - Daily (GHCND) dataset
                                  'locationid': 'FIPS:'+fips_id,  # Location using FIPS ID
                                  'datacategoryid': 'TEMP',  # Check for temperature data
                                  'limit': 1000  # Return maximum allowed
                              }).json()["results"]

    return pd.DataFrame(station_id)


def getdata(stationid, startyear, endyear):
    """
    This function grabs min and max daily temperature data for a chosen station from NOAA GHCND
    climate dataset. It then calculates the average temperature over a chosen date range.

    Parameters:
        - stationid:  A string of the GHCND station identifier (ex. GHCND:UK000000000)
        - startyear:  Integer of first year of date range (ex. 1900)
        - endyear:    Integer of last year of date range (ex. 2019)

    Returns:
        Pandas dataframe of daily mean temperatures (TAvg)
    """

    start = startyear
    end = endyear

    results = []

    while start <= end:

        currentstart = datetime.date(start, 1, 1)
        currentend = datetime.date(start + 1, 1, 1)

        # Update the cell with status information
        display.clear_output(wait=True)
        display.display(f'Gathering data for {str(currentstart)}')

        response = make_request('data',
                                {
                                    'datasetid': 'GHCND',
                                    # Global Historical Climatology Network - Daily (GHCND) dataset
                                    'datatypeid': ['TMAX', 'TMIN'],  # Max Temp
                                    'stationid': stationid,  # Station ID
                                    'startdate': currentstart,
                                    'enddate': currentend,
                                    'units': 'metric',  # Temperature units are degrees C
                                    'limit': 1000  # Return maximum allowed
                                })

        if response.ok:
            # We extend the list instead of appending to avoid getting a nested list
            results.extend(response.json()['results'])

        # Update the current date to avoid an infinite loop
        start += 1

    # Convert to Pandas Dataframe and tidy
    df = pd.DataFrame(results)
    del df["attributes"]  # Needless
    df["date"] = df["date"].str[:10]  # Get ride of time stamp
    df["date"] = df.apply(lambda x: datetime.datetime.strptime(x["date"], "%Y-%m-%d"),
                          axis=1)  # Convert dates to datetime format
    df = df.groupby("date").mean()  # Take mean of Tmax and Tmin
    df.rename(columns={"value": "TAvg"}, inplace=True)

    # Re-sample at chosen frequency and return dataframe
    return df[:-1]


def read_csv(name):
    df = pd.read_csv(name)
    # Convert dates to datetime format
    df["date"] = df.apply(lambda x: datetime.datetime.strptime(x["date"], "%Y-%m-%d"), axis=1)
    df.set_index(["date"], inplace=True)
    return df


def plot(df, startdate, enddate, timefreq, plot_line=True, savefig=False, figtitle='climate_stripes.png'):
    """
    This function resamples the daily temperature data at a chosen frequency and over the chosen time range.
    It then plots the corresponding climate stripes and time series.
    Warming stripes plot adapted from https://matplotlib.org/matplotblog/posts/warming-stripes/

    Parameters:
        - startdate:  String of starting date, e.g. '1955-01-01'
        - enddate:    String of end date, e.g. '1970-12-31'
        - timefreq: String of desired time frequency. For example, user can input 'yearly', 'monthly', or
                    'weekly'. A string using the pandas DateOffset alias can also be used. For documentation,
                    see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

    Returns:
        Plot of climate stripes and time series of mean temperatures (TAvg)
    """

    # Convert input time frequency to pandas data offset alias format
    if timefreq == 'yearly':
        samplefreq = 'AS'

    elif timefreq == 'monthly':
        samplefreq = 'MS'

    elif timefreq == 'weekly':
        samplefreq = 'W'

    else:
        samplefreq = timefreq

    # Resample data
    df = df[startdate: enddate].resample(samplefreq).mean()

    cmap = ListedColormap([
        '#08306b', '#08519c', '#2171b5', '#4292c6',
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
    ])

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_axes([0, 0, 1, 1])

    # Set limit values
    xmin = df.index[0].year
    xmax = df.index[-1].year
    ymin = int(df['TAvg'].min()) - 1
    ymax = int(df['TAvg'].max()) + 1

    # create a collection with a rectangle for each year
    col = PatchCollection([
        Rectangle((y, ymin), 1, ymax - ymin)
        for y in np.linspace(df.index[0].year, df.index[-1].year, len(df.index))
    ])

    # set data, colormap and color limits
    col.set_array(df["TAvg"])
    col.set_cmap(cmap)
    col.set_clim(ymin, ymax)
    ax.add_collection(col)

    if plot_line:
        # Plot time series
        ax.plot(np.linspace(xmin + (xmax - xmin) / (len(df.index)) / 2., xmax - (xmax - xmin) / (len(df.index)) / 2.,
                            len(df.index) - 1), df["TAvg"][:-1], color='yellow',
                marker='o', mec='#ffffcc', linewidth=4, markersize=16, zorder=1)

    # Set limits
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    if timefreq == 'yearly':
        ax.xaxis.set_ticks(np.arange(xmin, xmax + 1, 5))

    elif timefreq == 'monthly':
        ax.xaxis.set_ticks(np.arange(xmin, xmax + 1, 1))

    elif timefreq == 'weekly':
        ax.xaxis.set_ticks(np.arange(xmin, xmax + 1, 1))

    ax.yaxis.set_ticks(np.arange(ymin, ymax, 1))

    # Final touches
    cbar = plt.colorbar(col)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Temperature [degrees C]', fontsize=24)
    plt.xticks(rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_xlabel('Year', fontsize=24)
    ax.set_ylabel(u'Temperature [degree C]', fontsize=24)
    plt.title('Warming Stripes', fontsize=40, pad=20)

    # Save the figure if specified
    if savefig:
        plt.savefig(figtitle)

    # Show the figure if it is not saved
    else:
        plt.show()
