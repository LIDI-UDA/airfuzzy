import pickle

import lidi_fuzzy_control

if __name__ == '__main__':
    def calculate_mape(real, forecast):
        ape = []
        for i in range(len(real)):
            per_err = abs((real[i] - forecast[i]) / real[i])
            ape.append(per_err)
        MAPE = sum(ape) / len(ape)
        return MAPE


    sims = {
        's01': {'mf_type': 'trimf', 'defuzzify_method': 'centroid'},
        's02': {'mf_type': 'trimf', 'defuzzify_method': 'bisector'},
        's03': {'mf_type': 'trimf', 'defuzzify_method': 'mom'},
        's04': {'mf_type': 'trimf', 'defuzzify_method': 'som'},
        's05': {'mf_type': 'trimf', 'defuzzify_method': 'lom'},

        's06': {'mf_type': 'trapmf', 'defuzzify_method': 'centroid'},
        's07': {'mf_type': 'trapmf', 'defuzzify_method': 'bisector'},
        's08': {'mf_type': 'trapmf', 'defuzzify_method': 'mom'},
        's09': {'mf_type': 'trapmf', 'defuzzify_method': 'som'},
        's10': {'mf_type': 'trapmf', 'defuzzify_method': 'lom'},

        's11': {'mf_type': 'gaussmf', 'defuzzify_method': 'centroid'},
        's12': {'mf_type': 'gaussmf', 'defuzzify_method': 'bisector'},
        's13': {'mf_type': 'gaussmf', 'defuzzify_method': 'mom'},
        's14': {'mf_type': 'gaussmf', 'defuzzify_method': 'som'},
        's15': {'mf_type': 'gaussmf', 'defuzzify_method': 'lom'}
    }
    for k in sims.keys():
        sx = sims[k]
        df = lidi_fuzzy_control.lidi_fuzzy_sim(mf_type=sims[k]['mf_type'],
                                               defuzzify_method=sims[k]['defuzzify_method'])
        sims[k]['r1_mape'] = calculate_mape(df['r1_real_pm25'], df['r1_output_pm25'])
        sims[k]['r2_mape'] = calculate_mape(df['r2_real_o3'], df['r2_output_o3'])

    print(sims)
    with open('forecast_air.pkl', 'wb') as handle:
        pickle.dump(sims, handle, protocol=pickle.HIGHEST_PROTOCOL)
