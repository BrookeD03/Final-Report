import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('movies_metadata.csv')


features = ['budget', 'box_office', 'imdb_rating', 'rt_critic_score', 'rt_audience_score']
genre_cols = [col for col in df.columns if col.startswith('genre_')]
X = df[features + genre_cols]

X.fillna(X.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score for k={k}:', sil_score)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2')
plt.title('PCA Projection of Film Clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

db = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
df['dbscan_label'] = db.labels_

df['cluster'] = labels
df.to_csv('films_with_clusters.csv', index=False)
