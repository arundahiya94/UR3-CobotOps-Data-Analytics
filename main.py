from src.preprocessing import load_data, clean_data, normalize_data, apply_power_transform, remove_outliers
from src.clustering import kmeans_clustering, hierarchical_clustering, dbscan_clustering, plot_clusters
from src.utils import perform_pca
import pandas as pd

if __name__ == "__main__":
    # Load dataset
    file_path = 'data/raw/dataset.xlsx'
    df = load_data(file_path)

    # Preprocessing
    df = clean_data(df)
    columns_to_normalize = df.columns.difference(['grip_lost', 'Robot_ProtectiveStop'])
    df = apply_power_transform(df, columns_to_normalize)
    df = remove_outliers(df, columns_to_normalize)
    df = normalize_data(df, columns_to_normalize)

    # Dimensionality Reduction (PCA)
    data_pca = perform_pca(df)

    # K-Means Clustering
    clusters, silhouette = kmeans_clustering(data_pca, 3)
    print(f"K-Means Silhouette Score: {silhouette}")
    plot_clusters(data_pca, clusters, "K-Means Clusters")

    # Optional: Hierarchical or DBSCAN clustering can also be called here.