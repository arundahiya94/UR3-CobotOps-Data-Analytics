from sklearn.cluster import KMeans

def run_clustering(data, n_clusters=3):
    """Run KMeans clustering on the dataset."""
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    return clusters
