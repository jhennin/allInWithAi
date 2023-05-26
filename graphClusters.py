import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load embeddings from a CSV file
embeddings_csv_path = "data/all_in_tweets_embeddings.csv"
df = pd.read_csv(embeddings_csv_path, index_col=0)
embeddings = np.stack(df["embedding"].apply(eval).values)  # Convert embeddings from string to NumPy array

# Number of clusters
num_clusters = 2

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(embeddings)

# Assign cluster labels to each data point
df["cluster"] = cluster_labels

# Get a list of all tweet authors
authors = df["Name"].unique()

# Define colors for each author
colors = {
    "David": "blue",
    "Chamath": "red",
    "Friedberg": "purple",
    "Jason": "green"
}

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Define shapes for each cluster
shapes = ["o", "^"]  # Use circles for cluster 0 and squares for cluster 1

# Plot the clusters
plt.figure(figsize=(10, 8))
for i in range(num_clusters):
    for author in authors:
        cluster_points = embeddings_2d[(cluster_labels == i) & (df["Name"] == author)]
        color = colors.get(author, "black")  # Use black for unknown authors
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker=shapes[i], color=color, label=f"Cluster {i} ({author})")

plt.legend()
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("Clusters of Tweet Embeddings with Authors")
plt.show()
