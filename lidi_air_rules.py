import lidi_air_data as data
import lidi_crisp_sets as sets
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

df_fs = data.air_fuzzy_sets(drop_date=True,scale=True)
df = pd.get_dummies(df_fs)
df.to_csv("air_crispy_dummies.csv")

#df = data.normalized_data(scale=True)
#df.to_csv("normalized_data.csv")
#print(df)
