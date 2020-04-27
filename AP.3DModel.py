import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mpl_toolkits import mplot3d
%matplotlib notebook

df = pd.read_csv('ecounter.csv')
df_tomodel = df[['Position', 'Count', 'Sensor']]
df_tomodel = df_tomodel[df_tomodel != 'Null']
df_tomodel.dropna(inplace=True)
df_tomodel.reset_index(drop=True, inplace=True)
print('Shape', df_pollution_tomodel.shape)
df_tomodel[:10]


AffinityPropagation?


ap = AffinityPropagation(damping=0.90,
                        affinity='euclidean',
                         preference=-8000.0,
                         max_iter=3000)
preds = ap.fit_predict(df_tomodel)


ap.cluster_centers_


pd.DataFrame(ap.cluster_centers_, columns=df_tomodel.columns).T


df_tomodel['preds'] = preds


df_tomodel = df_tomodel.astype(float)


fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection="3d")
ax.scatter3D(df_tomodel['Position'], df_tomodel['Count'],  df_tomodel['Sensor'], c=preds, cmap='Accent')
plt.show()


cluster_centers_indices = ap.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % n_clusters_)



