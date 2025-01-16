import math

# https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
import numpy as np
import pandas as pd
from sklearn import preprocessing

usepa_airterms = ['good', 'moderate', 'unhealthy_groups',
                  'unhealthy', 'very_unhealthy', 'hazardous']
usepa_aircolors = ["#0ADD09", "#FFFF00", "#FF7E00", "#FF0000", "#99004C", "#7E0023"]

# https://www.researchgate.net/publication/316583277_Comparative_analysis_of_air_equivalent_-_effective_temperature_in_some_cities_of_Georgia_and_Brazil
airtmp_std_eet = ['sharply_coldly', 'coldly', 'moderately_coldly',
                  'comfortably', 'warmly', 'hotly']

    # https://www.rmets.org/metmatters/beaufort-scale
windspeed_beaufort = ['calm', 'light_air', 'light_breeze', 'gentle_breeze',
                      'moderate_breeze', 'fresh_breeze', 'strong_breeze',
                      'near_gale', 'gale', 'strong_gale', 'storm',
                      'violent_storm', 'hurricane']

# https://www.buildingscience.com/documents/reports/rr-0203-relative-humidity/view
rh_scales = ['uncomfortably_dry', 'confort_range', 'uncomfortably_humid']

# PAGASA Rain Rate Classification
# https://www.researchgate.net/publication/300307702_Circulation_types_classification_for_hourly_precipitation_events_in_Lublin_East_Poland/figures?lo=1&utm_source=google&utm_medium=organic
precip_scales = ['no_rain', 'light', 'moderate', 'heavy',
                 'intense', 'torrential']


class EnvironmentVar(object):
    # class attribute
    max_val = None
    bins = None
    bin_names = None
    df_name = None
    abbr = None
    name = None
    ud_step = None

    def __init__(self, name=None, abbr=None, df_name=None, names=None, bins=None, uni_s=0.0):
        self.name = name
        self.abbr = abbr
        self.df_name = df_name
        self.bin_names = names
        self.bins = bins
        self.ud_step = uni_s
        self.max_val = max(self.bins)

    def discretize_vals(self, vals):
        transformer = preprocessing.FunctionTransformer(pd.cut,
                                                        kw_args={'bins': self.bins,
                                                                 'labels': self.bin_names,
                                                                 'retbins': False})
        return transformer.fit_transform(vals)


CO = EnvironmentVar(name='Carbon monoxide', abbr='co', df_name='coppm',
                    names=usepa_airterms,
                    bins=[0, 4.5, 9.5, 12.5, 15.5, 30.5, 50.5], uni_s=0.5)
SO2 = EnvironmentVar(name='Sulfur dioxide', abbr='so2', df_name='so2ppb',
                     names=usepa_airterms,
                     bins=[0, 36, 76, 186, 305, 605, 1004], uni_s=1)
NO2 = EnvironmentVar(name='Nitrogen dioxide', abbr='no2', df_name='no2ppb',
                     names=usepa_airterms,
                     bins=[0, 54, 101, 361, 649, 1250, 2049], uni_s=1)
O3 = EnvironmentVar(name='Ozone', abbr='o3', df_name='o3ppm',
                    names=usepa_airterms,
                    bins=[0, 0.055, 0.071, 0.086, 0.106, 0.200, 0.300], uni_s=0.0001)  #
PM2_5 = EnvironmentVar(name='Particulate matter 2.5', abbr='pm2_5', df_name='pm2.5',
                       names=usepa_airterms,
                       bins=[0, 12.1, 35.5, 55.5, 150.5, 250.5, 500.5], uni_s=0.1)
AirTemp = EnvironmentVar(name='Temperature', abbr='airtemp', df_name='tempaire',
                         names=airtmp_std_eet,
                         bins=[0, 1, 9, 17, 23, 27, 42], uni_s=0.1)  #valle de la muerte
WindSpeed = EnvironmentVar(name='Wind speed', abbr='wind_speed', df_name='windspeed',
                           names=windspeed_beaufort,
                           bins=[0, 1, 6, 12, 20, 29, 38, 50, 62, 75, 89, 103, 118, 150], uni_s=0.1)
RH = EnvironmentVar(name='Relative humidity', abbr='rh', df_name='hr',
                    names=rh_scales,
                    bins=[0, 30, 60, 100], uni_s=1)


def ms2kmh(val):
    return val * 3.6


def ms2mph(val):
    return val * 2.23694


def ppm2ppb(val):
    return val * 1000


def eval_precip(val):
    # precip in mm
    if val == 0:
        return 'NO'
    else:
        return 'YES'


def eval_precip_mul(val):
    # precip in mm
    if val == 0:
        return precip_scales[0]
    elif 0 < val < 2.5:
        return precip_scales[1]
    elif 2.5 <= val < 7.5:
        return precip_scales[2]
    elif 7.5 <= val < 15:
        return precip_scales[3]
    elif 15 <= val < 30:
        return precip_scales[4]
    elif val >= 30:
        return precip_scales[5]
    else:
        return None


def eval_isweekend(val):
    if val:
        return 'YES'
    else:
        return 'NO'


def eval_dayHour(val):
    if 2 <= val < 6:
        return 'earlyMorning'
    elif 6 <= val < 12:
        return 'morning'
    elif 12 <= val < 18:
        return 'afternoon'
    elif 18 <= val < 24:
        return 'night'
    elif val <= 1:
        return 'night'
    else:
        return None
