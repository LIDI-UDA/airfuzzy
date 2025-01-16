import pickle
import pandas as pd

with open('forecast_air.pkl', 'rb') as handle:
    sims = pickle.load(handle)

resuls = []
for k in sims.keys():
    resuls.append(sims[k])

df = pd.DataFrame(resuls)
df = df.round(2)
print(df)

df_r1 = df[['mf_type', 'defuzzify_method', 'r1_mape']]
df_r2 = df[['mf_type', 'defuzzify_method', 'r2_mape']]
import plotly.express as px

fig = px.parallel_categories(df_r2, color='r2_mape',
                             #                             color_continuous_scale=px.colors.sequential.RdYiGn,
                             color_continuous_scale=px.colors.diverging.RdBu_r,
                             # title="Particulate Matter (PM2.5) forecast accuracy"
                             )
#fig.update_coloraxes(colorbar_ticklabelposition="outside top")
# fig.update_layout(coloraxis_colorbar_x=-0.1)
fig.update_layout(font_size=19)
#fig.update_traces(labelfont=dict(size=19, color="#000000", ), selector=dict(type='parcats'))
fig.update_traces(tickfont=dict(color="#0000FF"), selector=dict(type='parcats'))
fig.update_traces(line_shape='hspline', selector=dict(type='parcats'))
fig.update_layout(margin=dict(b=0, l=50, pad=0, r=0, t=30))
fig.update_traces(labelfont_size=24, selector=dict(type='parcats'))
fig.write_image("parcats_r2.pdf", scale=5)
fig.show()

