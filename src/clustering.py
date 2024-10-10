from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# K-Means Clustering with model, clusters, and silhouette score
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, clusters)
    return kmeans, clusters, silhouette_avg

# Hierarchical Clustering with model, clusters, and silhouette score
def hierarchical_clustering(data, n_clusters, linkage_method='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    clusters = model.fit_predict(data)
    silhouette_avg = silhouette_score(data, clusters)
    return model, clusters, silhouette_avg

# DBSCAN Clustering (No silhouette score due to possible noise points)
def dbscan_clustering(data, epsilon):
    dbscan = DBSCAN(eps=epsilon, min_samples=5)
    clusters = dbscan.fit_predict(data)
    return dbscan, clusters

# Function to plot clusters
def plot_clusters(data_pca, clusters, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters, palette='viridis')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Function to plot Elbow Method for optimal clusters
def plot_elbow(sse):
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal Clusters')
    plt.show()
