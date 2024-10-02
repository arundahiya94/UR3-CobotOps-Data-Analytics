from sklearn.metrics import silhouette_score

def evaluate_clustering(clusters, data):
    """Evaluate clustering performance using silhouette score."""
    score = silhouette_score(data, clusters)
    print(f'Silhouette Score: {score}')
