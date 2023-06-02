from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import datasets
from  matplotlib import pyplot as plt
import pandas as pd
import numpy as np

iris_data = datasets.load_iris()
iris_source= scale(iris_data.data)
iris_target = pd.DataFrame(iris_data.target)
type(iris_target)

iris_variable = iris_data.feature_names
iris_df= pd.DataFrame(iris_data.data)
iris_df.columns=[name.replace(" (cm)","").replace(' ','_') for name in iris_data.feature_names]
iris_target.columns=['target']
iris_df

Kmeans_clusting = KMeans(n_clusters=3,random_state=5)

Kmeans_clusting.fit(iris_source)


relabel = np.choose(Kmeans_clusting.labels_,[2,0,1])
plt.subplot(2,2,1)
plt.scatter(x=iris_df.sepal_length,y=iris_df.sepal_width,c=iris_data.target)
plt.subplot(2,2,2)
plt.scatter(x=iris_df.sepal_length,y=iris_df.sepal_width,c=relabel)
plt.subplot(2,2,3)
plt.scatter(x=iris_df.petal_length,y=iris_df.petal_width,c=iris_data.target)
plt.subplot(2,2,4)
plt.scatter(x=iris_df.petal_length,y=iris_df.petal_width,c=relabel)