from src.preprocessing import load_data, clean_data, normalize_data, apply_power_transform, remove_outliers
from src.clustering import kmeans_clustering, hierarchical_clustering, dbscan_clustering, plot_clusters
from src.utils import perform_pca
import joblib
import pandas as pd
import os
import traceback
import logging

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'project_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Optional: Save models to a file
def save_model(model, model_name, folder="models/"):
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save the model as .pkl
    model_file_path = os.path.join(folder, f"{model_name}.pkl")
    joblib.dump(model, model_file_path)
    logging.info(f"Model saved as {model_file_path}")

if __name__ == "__main__":
    logging.info("Starting the project execution.")
    logging.info(f"Current working directory: {os.getcwd()}")
    try:
        # Load dataset
        file_path = 'D:/git/UR3-CobotOps-Data-Analytics/data/dataset.xlsx'
        try:
            df = load_data(file_path)
            logging.info(f"Data loaded from {file_path}")
        except FileNotFoundError as fnf_error:
            logging.error(f"File not found: {file_path}")
            raise fnf_error
        except Exception as load_error:
            logging.error(f"Error loading data from {file_path}: {str(load_error)}")
            raise load_error

        # Preprocessing
        try:
            df = clean_data(df)
            columns_to_normalize = df.columns.difference(['grip_lost', 'Robot_ProtectiveStop'])

            # Apply power transform
            df = apply_power_transform(df, columns_to_normalize)
            
            # Remove outliers
            df = remove_outliers(df, columns_to_normalize)
            
            # Normalize data
            df = normalize_data(df, columns_to_normalize)
            logging.info("Data preprocessed (power transform, outlier removal, normalization).")
        except KeyError as key_error:
            logging.error(f"Column not found during preprocessing: {str(key_error)}")
            raise key_error
        except Exception as preprocessing_error:
            logging.error("Error during data preprocessing.")
            raise preprocessing_error

        # Dimensionality Reduction (PCA)
        try:
            data_pca = perform_pca(df)
        except ValueError as pca_error:
            logging.error("Error during PCA dimensionality reduction.")
            raise pca_error
        except Exception as pca_exception:
            logging.error(f"An unexpected error occurred during PCA: {str(pca_exception)}")
            raise pca_exception

        # K-Means Clustering
        try:
            kmeans_model, clusters, silhouette = kmeans_clustering(data_pca, n_clusters=3)
            logging.info(f"K-Means clustering completed. Silhouette Score: {silhouette}")
        except ValueError as clustering_error:
            logging.error(f"Error during K-Means clustering: {str(clustering_error)}")
            raise clustering_error
        except Exception as clustering_exception:
            logging.error("An unexpected error occurred during K-Means clustering.")
            raise clustering_exception
        
        # Plot K-Means clusters
        try:
            plot_clusters(data_pca, clusters, "K-Means Clusters")
            logging.info("K-Means clusters plotted.")
        except Exception as plot_error:
            logging.error(f"Error during plotting: {str(plot_error)}")
            raise plot_error

        # Save the K-Means model
        try:
            save_model(kmeans_model, model_name="kmeans_model")
        except Exception as save_error:
            logging.error(f"Error saving the model: {str(save_error)}")
            raise save_error

        # Optional: Hierarchical or DBSCAN clustering can also be called here
        # Uncomment and use similar error handling as above for additional clustering methods
        # For example:
        # try:
        #     hierarchical_model, hier_clusters, hier_silhouette = hierarchical_clustering(data_pca, n_clusters=3)
        #     print(f"Hierarchical Clustering Silhouette Score: {hier_silhouette}")
        #     plot_clusters(data_pca, hier_clusters, "Hierarchical Clusters")
        #     save_model(hierarchical_model, model_name="hierarchical_model")
        # except Exception as hierarchical_error:
        #     print(f"Error during Hierarchical clustering: {str(hierarchical_error)}")
        #     raise hierarchical_error

        # try:
        #     dbscan_model, dbscan_clusters, dbscan_silhouette = dbscan_clustering(data_pca, eps=0.5, min_samples=5)
        #     print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
        #     plot_clusters(data_pca, dbscan_clusters, "DBSCAN Clusters")
        #     save_model(dbscan_model, model_name="dbscan_model")
        # except Exception as dbscan_error:
        #     print(f"Error during DBSCAN clustering: {str(dbscan_error)}")
        #     raise dbscan_error

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        traceback.print_exc()  # Provides a detailed stack trace for debugging purposes
