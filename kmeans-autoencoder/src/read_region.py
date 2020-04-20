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

NOTES ON MISSING DATA:
  All data returned by this function are NumPy masked arrays. Areas with
  missing data are "masked" (i.e., masking is True). You can determine where
  an array is masked using np.ma.getmaskarray:

  >>> dat = read_region('natlantic2_goes16_2017_306_1425.nc',retrievals=True)

  >>> np.ma.getmaskarray(dat['cth'])
  array([[False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         ...,
         [ True,  True,  True, ...,  True,  True,  True],
         [False, False,  True, ...,  True,  True,  True],
         [False, False, False, ...,  True,  True, False]])

  Determine if a specific element is masked:
      
  >>> dat['cth'][0,0] is np.ma.masked
  False

  If you don't want to deal to with masked arrays, you can replace any masked
  values with a fill value of your choice:

  >>> dat['cth'].filled(-999)
  array([[ 8633.991 ,  8635.822 ,  8636.128 , ..., 11019.013 , 10778.222 ,
          10817.286 ],
         [ 8565.63  ,  8604.693 ,  8656.27  , ..., 10271.615 , 10294.504 ,
          10876.491 ],
         [ 8569.598 ,  8771.63  ,  8738.976 , ..., 10450.148 , 10382.702 ,
          10423.597 ],
         ...,
         [ -999.    ,  -999.    ,  -999.    , ...,  -999.    ,  -999.    ,
           -999.    ],
         [  941.4956,  1016.5711,  -999.    , ...,  -999.    ,  -999.    ,
           -999.    ],
         [ 1183.8125,  1549.1191,  1199.9874, ...,  -999.    ,  -999.    ,
            698.8734]], dtype=float32)

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
