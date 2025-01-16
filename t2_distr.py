import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
df_ori = pd.read_excel("contaminantes-meteorologia-1anio.xlsx")
df_ori = pd.DataFrame(df_ori[['EC_TIME_STAMP', 'TEMPAIRE_AV', 'O3', 'CO', 'NO2', 'SO2', 'PM2_5']])
df_ori['Month'] = df_ori['EC_TIME_STAMP'].dt.month
df_ori = df_ori.replace(0, np.nan)
df_ori.dropna(inplace=True)
df_ori = df_ori[df_ori['Month'] == 10]
df_ori['Hour'] = df_ori['EC_TIME_STAMP'].dt.hour
df_ori.rename({'TEMPAIRE_AV': 'TEMP'}, axis=1, inplace=True)
# print(df_ori.shape)

#fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=False, sharex=False)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False, sharex=False)
plt.subplots_adjust(wspace=0.4)

df_ori['PM2_5'] = df_ori['PM2_5'].round(2)
g1 = sns.boxplot(ax=axes[0], y=df_ori['PM2_5'])
axes[0].title.set_text("a) Boxplot: outliers")

vals = df_ori['PM2_5'].drop_duplicates().sort_values()
vals_diff = pd.DataFrame(np.diff(vals), columns=['diff'])
# sns.boxplot(ax=axes[1], y=df_ori['PM2_5'])
g2 = sns.histplot(ax=axes[1], data=vals_diff, x='diff', kde=False, bins=5)
g2.set(xlabel=None)
axes[1].title.set_text("b) Histogram of intervals")

fig.savefig('boxplot_sequences.pdf', bbox_inches='tight', dpi=600)
plt.show()
