import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/Mtech/AILab/iris")

df=data[["PetalLengthCm","PetalWidthCm"]]
df.head()

import matplotlib.pyplot as plt
plt.scatter(df["PetalLengthCm"],df["PetalWidthCm"])
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.show()

from sklearn.cluster import KMeans

km=KMeans(n_clusters=3)
y_pred=km.fit_predict(data[["PetalLengthCm","PetalWidthCm"]])
y_pred
df["cluster"]=y_pred
print(df)
# Plotting the clusters
plt.scatter(df["PetalLengthCm"][df["cluster"]==0], df["PetalWidthCm"][df["cluster"]==0], color='green', label ='Cluster 1')
plt.scatter(df["PetalLengthCm"][df["cluster"]==1], df["PetalWidthCm"][df["cluster"]==1], color='red', label ='Cluster 2')
plt.scatter(df["PetalLengthCm"][df["cluster"]==2], df["PetalWidthCm"][df["cluster"]==2], color='blue', label ='Cluster 3')
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Scatter plot for each cluster
plt.scatter(df["PetalLengthCm"][df["cluster"]==0], df["PetalWidthCm"][df["cluster"]==0], color='green', label='Cluster 1')
plt.scatter(df["PetalLengthCm"][df["cluster"]==1], df["PetalWidthCm"][df["cluster"]==1], color='red', label='Cluster 2')
plt.scatter(df["PetalLengthCm"][df["cluster"]==2], df["PetalWidthCm"][df["cluster"]==2], color='blue', label='Cluster 3')

# Plot cluster centers
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            s=200, marker='*', color='black', label='Centroids')

plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.legend()
plt.show()
