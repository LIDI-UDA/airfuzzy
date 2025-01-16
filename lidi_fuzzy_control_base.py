import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
# sns.set_palette(sns.color_palette("Paired"))
# sns.set_palette("dark")

colors = ["#0ADD09", "#FFFF00", "#FF7E00", "#FF0000", "#99004C", "#7E0023"]
sns.set_palette(sns.color_palette(colors))

df_ori = pd.read_excel("contaminantes-meteorologia-1anio.xlsx")
df_ori = pd.DataFrame(df_ori[['EC_TIME_STAMP', 'TEMPAIRE_AV', 'O3', 'CO', 'NO2', 'SO2', 'PM2_5']])
df_ori['Month'] = df_ori['EC_TIME_STAMP'].dt.month
df_ori = df_ori.replace(0, np.nan)
df_ori.dropna(inplace=True)
df_ori = df_ori[df_ori['Month'] == 10]
df_ori['Hour'] = df_ori['EC_TIME_STAMP'].dt.hour
df_ori.rename({'TEMPAIRE_AV': 'TEMP'}, axis=1, inplace=True)
df_ori = df_ori.round(2)

# New Antecedent/Consequent objects hold universe variables and membership
# functions
names = ['good', 'moderate', 'unhealthy_groups', 'unhealthy', 'very_unhealthy', 'hazardous']

pm2_5 = ctrl.Consequent(np.arange(0, 500, 0.4), 'pm2_5')
pm2_5[names[0]] = fuzz.trimf(pm2_5.universe, [0, 0, 12])  # 0,1,0
pm2_5[names[1]] = fuzz.trimf(pm2_5.universe, [6, 12, 35.5])  # 0,1,0
pm2_5[names[2]] = fuzz.trimf(pm2_5.universe, [23.75, 35.5, 55.5])  # 0,1,0
pm2_5[names[3]] = fuzz.trimf(pm2_5.universe, [45.50, 55.5, 150.5])  # 0,1,0
pm2_5[names[4]] = fuzz.trimf(pm2_5.universe, [103.0, 150.5, 250.5])  # 0,1,0
pm2_5[names[5]] = fuzz.trimf(pm2_5.universe, [200.5, 500, 500])  # 0,1,0

co = ctrl.Antecedent(np.arange(0, 50.5, 0.1), 'co')
co[names[0]] = fuzz.trimf(co.universe, [0, 0, 4.5])  # 0,1,0
co[names[1]] = fuzz.trimf(co.universe, [2.25, 4.5, 9.5])  # 0,1,0
co[names[2]] = fuzz.trimf(co.universe, [7.00, 9.5, 12.5])  # 0,1,0
co[names[3]] = fuzz.trimf(co.universe, [11, 12.5, 15.5])  # 0,1,0
co[names[4]] = fuzz.trimf(co.universe, [14, 15.5, 30.5])  # 0,1,0
co[names[5]] = fuzz.trimf(co.universe, [23, 50.5, 50.5])  # 0,1,0

so2 = ctrl.Antecedent(np.arange(0, 1004, 0.5), 'so2')
so2[names[0]] = fuzz.trimf(so2.universe, [0, 0, 35])  # 0,1,0
so2[names[1]] = fuzz.trimf(so2.universe, [17.5, 36, 75])  # 0,1,0
so2[names[2]] = fuzz.trimf(so2.universe, [55.5, 76, 185])  # 0,1,0
so2[names[3]] = fuzz.trimf(so2.universe, [130.5, 186, 304])  # 0,1,0
so2[names[4]] = fuzz.trimf(so2.universe, [245, 305, 604])  # 0,1,0
so2[names[5]] = fuzz.trimf(so2.universe, [454.5, 605, 1004])  # 0,1,0

no2 = ctrl.Antecedent(np.arange(0, 2049, 0.5), 'no2')
no2[names[0]] = fuzz.trimf(no2.universe, [0, 0, 53])  # 0,1,0
no2[names[1]] = fuzz.trimf(no2.universe, [26.5, 54, 100])  # 0,1,0
no2[names[2]] = fuzz.trimf(no2.universe, [77, 101, 360])  # 0,1,0
no2[names[3]] = fuzz.trimf(no2.universe, [230.5, 361, 649])  # 0,1,0
no2[names[4]] = fuzz.trimf(no2.universe, [505, 650, 1249])  # 0,1,0
no2[names[5]] = fuzz.trimf(no2.universe, [949.5, 1250, 2049])  # 0,1,0

pm2_5.view()
co.view()
so2.view()
no2.view()

rule1 = ctrl.Rule(so2[names[2]], pm2_5[names[3]])

tipping_ctrl = ctrl.ControlSystem([rule1])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# 2022-03-17 07:55, 0.891
tipping.input['so2'] = 71
tipping.compute()
print(tipping.output['pm2_5'])
co.view(sim=tipping)
plt.show()
