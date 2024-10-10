from sklearn.metrics import silhouette_score

def evaluate_kmeans(kmeans, data):
    return silhouette_score(data, kmeans.labels_)

def evaluate_dbscan(dbscan, data):
    return silhouette_score(data, dbscan.labels_)
