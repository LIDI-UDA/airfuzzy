import math
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns
import lidi_air_data
import lidi_crisp_sets as sets

sns.set()
sns.set_palette(sns.color_palette(sets.usepa_aircolors))
air_data = lidi_air_data.normalized_data(scale=True)


def make_trapmf(var_ctrl, bins, names, pct=0.1):
    for i, _ in enumerate(bins):
        if i < len(bins) - 1:
            pct_mfs = (bins[i + 1] - bins[i]) * pct
            if i == 0:
                points = [bins[i], bins[i],
                          bins[i + 1] - pct_mfs, bins[i + 1]]
            elif i == len(bins) - 2:
                points = [bins[i] - pct_mfs, bins[i] + pct_mfs,
                          bins[i + 1], bins[i + 1]]
            else:
                points = [bins[i] - pct_mfs, bins[i] + pct_mfs,
                          bins[i + 1] - pct_mfs, bins[i + 1] + pct_mfs]
            var_ctrl[names[i]] = fuzz.trapmf(var_ctrl.universe, points)


def make_trimf(var_ctrl, bins, names, pct=0.1):
    for i, _ in enumerate(bins):
        if i < len(bins) - 1:
            pct_mfs = (bins[i + 1] - bins[i]) * pct
            if i == 0:
                points = [bins[i], bins[i], bins[i + 1] + pct_mfs]
            elif i == len(bins) - 2:
                points = [bins[i] - pct_mfs, bins[i + 1], bins[i + 1]]
            else:
                middle = bins[i] + ((bins[i + 1] - bins[i]) / 2)
                points = [bins[i] - pct_mfs, middle, bins[i + 1] + pct_mfs]
            var_ctrl[names[i]] = fuzz.trimf(var_ctrl.universe, points)


def make_gaussmf(var_ctrl, bins, names, data=0):
    for i, _ in enumerate(bins):
        band_fail_mean = False
        if i < len(bins) - 1:
            col = ''
            if var_ctrl.label == sets.PM2_5.abbr:
                col = sets.PM2_5.df_name
            elif var_ctrl.label == sets.CO.abbr:
                col = sets.CO.df_name
            elif var_ctrl.label == sets.O3.abbr:
                col = sets.O3.df_name
            elif var_ctrl.label == sets.WindSpeed.abbr:
                col = sets.WindSpeed.df_name
            elif var_ctrl.label == sets.AirTemp.abbr:
                col = sets.AirTemp.df_name

            bin_data = air_data[col][
                (air_data[col] >= bins[i]) & (air_data[col] <= bins[i + 1])]
            try:
                bin_mean = bin_data.mean()
                bin_std = bin_data.std()
                if bin_mean == 0 or math.isnan(bin_mean):
                    bin_mean = bins[i] + ((bins[i + 1] - bins[i]) / 2)
                    band_fail_mean = True
                if bin_std == 0 or math.isnan(bin_std) or band_fail_mean:
                    bin_std = bin_mean * 0.4
            except:
                bin_mean = bins[i] + ((bins[i + 1] - bins[i]) / 2)
                bin_std = bin_mean * 0.4
            print(col, bin_mean, bin_std, len(bin_data), names[i])
            var_ctrl[names[i]] = fuzz.gaussmf(var_ctrl.universe, bin_mean, bin_std)


def construct_mf(mf_type='trimf', var_ctrl=None, bins=None, names=None, pct=0.1):
    if mf_type == 'trimf':
        make_trimf(var_ctrl=var_ctrl, bins=bins, names=names, pct=pct)
    elif mf_type == 'trapmf':
        make_trapmf(var_ctrl=var_ctrl, bins=bins, names=names, pct=pct)
    elif mf_type == 'gaussmf':
        make_gaussmf(var_ctrl=var_ctrl, bins=bins, names=names)
    else:
        print("Wrong mf_type...")


################################################## CONTROL SIMULATION

def lidi_fuzzy_sim(mf_type='trimf', defuzzify_method='centroid', pct_memb=0.1):
    # centroid, bisector, mom, som, lom
    co = ctrl.Antecedent(np.arange(0, sets.CO.max_val, sets.CO.ud_step), sets.CO.abbr)
    air_temp = ctrl.Antecedent(np.arange(0, sets.AirTemp.max_val, sets.AirTemp.ud_step),
                               sets.AirTemp.abbr)
    wind_speed = ctrl.Antecedent(np.arange(0, sets.WindSpeed.max_val, sets.WindSpeed.ud_step),
                                 sets.WindSpeed.abbr)
    o3 = ctrl.Consequent(np.arange(0, sets.O3.max_val, sets.O3.ud_step), sets.O3.abbr,
                         defuzzify_method=defuzzify_method)
    pm2_5 = ctrl.Consequent(np.arange(0, sets.PM2_5.max_val, sets.PM2_5.ud_step), sets.PM2_5.abbr,
                            defuzzify_method=defuzzify_method)

    construct_mf(var_ctrl=pm2_5, mf_type=mf_type,
                 bins=sets.PM2_5.bins, names=sets.PM2_5.bin_names, pct=pct_memb)
    construct_mf(var_ctrl=co, mf_type=mf_type,
                 bins=sets.CO.bins, names=sets.CO.bin_names, pct=pct_memb)
    construct_mf(var_ctrl=o3, mf_type=mf_type,
                 bins=sets.O3.bins, names=sets.O3.bin_names, pct=pct_memb)
    construct_mf(var_ctrl=wind_speed, mf_type=mf_type,
                 bins=sets.WindSpeed.bins, names=sets.WindSpeed.bin_names, pct=pct_memb)
    construct_mf(var_ctrl=air_temp, mf_type=mf_type,
                 bins=sets.AirTemp.bins, names=sets.AirTemp.bin_names, pct=pct_memb)

    rule1 = ctrl.Rule(co[sets.SO2.bin_names[0]],
                      pm2_5[sets.PM2_5.bin_names[0]])

    rule2_1 = ctrl.Rule(air_temp[sets.AirTemp.bin_names[4]] &
                        wind_speed[sets.WindSpeed.bin_names[0]],
                        o3[sets.PM2_5.bin_names[2]])
    rule2_2 = ctrl.Rule(air_temp[sets.AirTemp.bin_names[5]] &
                        wind_speed[sets.WindSpeed.bin_names[1]],
                        o3[sets.PM2_5.bin_names[3]])

    tipping_ctrl = ctrl.ControlSystem([rule1, rule2_1, rule2_2])
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    # 24_01
    test_co = np.array([1.326, 1.379, 1.334, 1.358, 1.343, 1.329])
    real_pm25 = np.array([5.033, 2.743, 2.927, 5.028, 1.723, 0.827])

    test_airtemp = np.array([33.77531807, 39.70076336, 40.44529262, 41.47251908, 40.35292621, 39.09338422])
    test_windspe = np.array([1.667, 1.483, 1.383, 2.367, 2.117, 2.283])

    real_o3 = np.array([0.073170732, 0.092682927, 0.092682927, 0.07804878, 0.087804878, 0.073170732])

    output_pm25 = []
    output_o3 = []

    for v1, v2, v3 in zip(test_co, test_airtemp, test_windspe):
        tipping.input[sets.CO.abbr] = v1
        tipping.input[sets.AirTemp.abbr] = v2
        tipping.input[sets.WindSpeed.abbr] = v3
        tipping.compute()

        tipping.output['pm2_5']
        plt.title("pm2_5 " + mf_type + "-" + defuzzify_method)
        pm2_5.view(sim=tipping)

        tipping.output['o3']
        plt.title("o3 " + mf_type + "-" + defuzzify_method)
        o3.view(sim=tipping)

        output_pm25.append(tipping.output[sets.PM2_5.abbr])
        output_o3.append(tipping.output[sets.O3.abbr])

    # mse_1 = mean_squared_error(real_pm25, output_pm25)
    # mse_2 = mean_squared_error(real_o3, output_o3)
    # print("mse_r1:", mse_1, "rmse_1:", sqrt(mse_1))
    # print("mse_r2:", mse_2, "rmse_2:", sqrt(mse_2))

    o3.view(sim=tipping)
    plt.show()
    df_pred = pd.DataFrame(list(zip(real_pm25, output_pm25, real_o3, output_o3)),
                           columns=['r1_real_pm25', 'r1_output_pm25', 'r2_real_o3', 'r2_output_o3'])
    return df_pred


if __name__ == '__main__':
    _ = lidi_fuzzy_sim()
