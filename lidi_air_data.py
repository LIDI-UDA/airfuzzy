import pandas as pd
from sklearn import preprocessing

import lidi_crisp_sets as air_fs
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

sns.set()


def read_df():
    from datetime import datetime
    custom_date_parser = lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M")
    df = pd.read_csv("IDataTransformacion-2021.csv",
                     sep=";",
                     parse_dates=[0],
                     date_parser=custom_date_parser
                     )
    return df


from sklearn.preprocessing import FunctionTransformer
import numpy as np


def normalized_data(scale=False):
    df = read_df()
    df.drop(['winddir', 'patm', 'radglobal'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['so2ppb'] = df['so2ppm'].apply(air_fs.ppm2ppb)
    df['no2ppb'] = df['no2ppm'].apply(air_fs.ppm2ppb)
    df['hour'] = df['fecha'].dt.hour
    # df['day'] = df['fecha'].dt.day_name()
    df['isweekend'] = df['fecha'].dt.weekday > 4
    df.drop(['o3', 'so2', 'co', 'no2'], axis=1, inplace=True)
    df.drop(['so2ppm', 'no2ppm'], axis=1, inplace=True)
    if scale:
        s = {'so2ppb': 1004, 'no2ppb': 2049, 'o3ppm': 0.200, 'coppm': 50.5, 'pm2.5': 500.5,
             'tempaire': 42, 'windspeed': 150}
        for v in list(s):
            # pasar a distribucion normal
            qt = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
            # df[v] = qt.fit_transform(df[[v]])
            # pasar a scalas de la EPA
            mmscaler = preprocessing.MinMaxScaler(feature_range=(0, s[v]))
            df[v] = mmscaler.fit_transform(df[[v]])
            del qt, mmscaler
    return df


def air_fuzzy_sets(drop_date=True, scale=False):
    df = normalized_data(scale=scale)
    df['o3ppm'] = air_fs.O3.discretize_vals(df['o3ppm'])
    df['coppm'] = air_fs.CO.discretize_vals(df['coppm'])
    df['so2ppb'] = air_fs.SO2.discretize_vals(df['so2ppb'])
    df['no2ppb'] = air_fs.NO2.discretize_vals(df['no2ppb'])
    df['pm2.5'] = air_fs.PM2_5.discretize_vals(df['pm2.5'])
    df['tempaire'] = air_fs.AirTemp.discretize_vals(df['tempaire'])
    df['windspeed'] = air_fs.WindSpeed.discretize_vals(df['windspeed'])
    df['hr'] = air_fs.RH.discretize_vals(df['hr'])
    df['precip'] = df['precip'].apply(air_fs.eval_precip)
    df['hour'] = df['hour'].apply(air_fs.eval_dayHour)
    df['isweekend'] = df['isweekend'].apply(air_fs.eval_isweekend)
    if drop_date:
        df.drop(['fecha'], axis=1, inplace=True)
    return df


def day_var_plot(date=None, var=None):
    df = normalized_data()
    ax = sns.lineplot(data=df[df['fecha'].dt.strftime('%Y-%m-%d') == '2021-01-26'],
                      x="fecha",
                      y="coppm")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.xticks(rotation=45)
    plt.show()
    print(df.columns)
    return


if __name__ == '__main__':
    df = air_fuzzy_sets(scale=True)
    s = {'so2ppb': 1004, 'no2ppb': 2049, 'o3ppm': 0.200, 'coppm': 50.5, 'pm2.5': 500.5}
    for v in list(s):
        pd.value_counts(df[v]).plot.bar()

        plt.title('true-' + str(v))
        plt.show()
    # print(air_fuzzy_sets().to_csv("air.csv"))
    # data = air_fuzzy_sets()
    # print(data[data.precip != 'no_rain'])
