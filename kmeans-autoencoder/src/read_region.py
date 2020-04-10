# -*- coding: utf-8 -*-

"""
Read extracted GOES ABI data and retrievals

INPUTS:
  fin        data file name

KEYWORDS:
  channels     set to True to return ABI channel data
  geo          set to True to return geolocation data
  retrievals   set to True to return retrieval data

RETURNS:
  A dictionary containing the requested data, specified by setting at least
  one of the keywords to True: channels, geo, retrievals

EXAMPLE:

  Read reflectances/brightness temperatures for all 16 channels

  >>> dat = read_region('natlantic_goes16_2017_300_1600.nc', channels=True)

VARIABLES THAT MAY BE IN THE RETURNED DICTIONARY
  channels data:
    c00    channel data for all 16 ABI channels; shape is (nlat,nlon,16)

  geo data:
    lat    latitude in degrees
    lon    longitude in degrees

  retrievals data:
    cmask    cloud mask
    cphase   cloud phase
    cth      cloud top height (meters)
    ctype    cloud type
    cwp      cloud water path (g/m^3)
    reff     cloud effective radius (micrometers)
    sza      solar zenith angle (degrees)
    tau      optical depth

MODIFICATION HISTORY
2020/04/05  Written by John Haynes (john.haynes@colostate.edu)
"""

from netCDF4 import Dataset

def read_region(fin,channels=False,geo=False,retrievals=False):
    dat = {}
    dataset = Dataset(fin)

    if channels:
        dat.update({'c00': dataset.variables['c00'][:]})

    if geo:
        dat.update({'lat': dataset.variables['lat'][:],
                    'lon': dataset.variables['lon'][:]})

    if retrievals:
        dat.update({'cmask': dataset.variables['cmask'][:],
                    'cphase': dataset.variables['cphase'][:],
                    'cth': dataset.variables['cth'][:],
                    'ctype': dataset.variables['ctype'][:],
                    'cwp': dataset.variables['cwp'][:],
                    'reff': dataset.variables['reff'][:],
                    'sza': dataset.variables['sza'][:],
                    'tau': dataset.variables['tau'][:]})

    dataset.close()

    return(dat)
