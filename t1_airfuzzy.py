import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
# sns.set_palette(sns.color_palette("Paired"))
# sns.set_palette("dark")

colors = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
# colors = ["green", "yellow", "orange", "red", "purple", "maroon"]
sns.set_palette(sns.color_palette(colors))

df_ori = pd.read_excel("contaminantes-meteorologia-1anio.xlsx")
df_ori = pd.DataFrame(df_ori[['EC_TIME_STAMP', 'TEMPAIRE_AV', 'O3', 'CO', 'NO2', 'SO2', 'PM2_5']])
df_ori['Month'] = df_ori['EC_TIME_STAMP'].dt.month
df_ori = df_ori.replace(0, np.nan)
df_ori.dropna(inplace=True)
df_ori = df_ori[df_ori['Month'] == 10]
df_ori['Hour'] = df_ori['EC_TIME_STAMP'].dt.hour
df_ori.rename({'TEMPAIRE_AV': 'TEMP'}, axis=1, inplace=True)
# print(df_ori.shape)

# New Antecedent/Consequent objects hold universe variables and membership
# functions
df_ori['PM2_5'] = df_ori['PM2_5'].round(2)
vals_ori = df_ori['PM2_5'].drop_duplicates().sort_values()
vals_diff = pd.DataFrame(np.diff(vals_ori))
print(vals_diff.describe())

max_val_ori = vals_ori.max()
print(max_val_ori)
vals_seq = np.arange(0, 350, 0.001)

# problem typing

names = ['good', 'moderate', 'unhealthy_groups', 'unhealthy', 'very_unhealthy', 'hazardous']

make_auto_trian = False
pm2_5_o = ctrl.Antecedent(vals_ori, 'pm2_5')
pm2_5_s = ctrl.Antecedent(vals_seq, 'pm2_5')
if make_auto_trian:
    # Auto-membership function population is possible with .automf(3, 5, or 7)
    pm2_5_o.automf(names=names)
    pm2_5_s.automf(names=names)
else:
    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    pm2_5_o[names[0]] = fuzz.trimf(pm2_5_o.universe, [0, 0, 15.4])  # 0,1,0
    pm2_5_o[names[1]] = fuzz.trimf(pm2_5_o.universe, [7, 15.4, 40.4])  # 0,1,0
    pm2_5_o[names[2]] = fuzz.trimf(pm2_5_o.universe, [15.4, 40.4, 65.4])  # 0,1,0
    pm2_5_o[names[3]] = fuzz.trimf(pm2_5_o.universe, [40.4, 65.4, 150.4])  # 0,1,0
    pm2_5_o[names[4]] = fuzz.trimf(pm2_5_o.universe, [65.4, 150.4, 250.4])  # 0,1,0
    pm2_5_o[names[5]] = fuzz.trimf(pm2_5_o.universe, [250.4, 350, 350])  # 0,1,0

    pm2_5_s[names[0]] = fuzz.trimf(pm2_5_s.universe, [0, 0, 15.4])  # 0,1,0
    pm2_5_s[names[1]] = fuzz.trimf(pm2_5_s.universe, [12, 15.4, 40.4])  # 0,1,0
    pm2_5_s[names[2]] = fuzz.trimf(pm2_5_s.universe, [15.4, 40.4, 65.4])  # 0,1,0
    pm2_5_s[names[3]] = fuzz.trimf(pm2_5_s.universe, [40.4, 65.4, 150.4])  # 0,1,0
    pm2_5_s[names[4]] = fuzz.trimf(pm2_5_s.universe, [65.4, 150.4, 250.4])  # 0,1,0
    pm2_5_s[names[5]] = fuzz.trimf(pm2_5_s.universe, [250.4, 350, 350])  # 0,1,0

# pm2_5.view()
fig_o, ax_o = pm2_5_s.ret_view()
fig_o.set_size_inches(5.5, 3.5)

fig_s, ax_s = pm2_5_s.ret_view()
fig_s.set_size_inches(5.5, 3.5)

#ax_o.get_legend().remove()
#fig_o.savefig('crisp_definition_well.pdf', bbox_inches='tight', dpi=600)
#plt.show()

fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False, sharex=False)
plt.subplots_adjust(wspace=0.4)
ax_o.plot()
plt.show()
#axes[1] = fig_s
#fig.savefig('issue_crisp_definition.pdf', bbox_inches='tight', dpi=600)
