import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lidi_air_data as data

sns.set()

df_ori = data.normalized_data(scale=True)
# fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=False, sharex=False)
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False, sharex=False)
plt.subplots_adjust(wspace=0.4)

df_ori['pm2.5'] = df_ori['pm2.5'].round(2)
g1 = sns.boxplot(ax=axes[0], y=df_ori['pm2.5'])
axes[0].title.set_text("a) Boxplot: outliers")

vals = df_ori['pm2.5'].drop_duplicates().sort_values()
vals_diff = pd.DataFrame(np.diff(vals), columns=['diff'])
# sns.boxplot(ax=axes[1], y=df_ori['PM2_5'])
g2 = sns.histplot(ax=axes[1], data=vals_diff, x='diff', kde=False, bins=5)
g2.set(xlabel=None)
axes[1].title.set_text("b) Histogram of intervals")

fig.savefig('boxplot_sequences.pdf', bbox_inches='tight', dpi=600)
plt.show()
print(df_ori['pm2.5'].describe())