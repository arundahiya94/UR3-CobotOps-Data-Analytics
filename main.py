from src.preprocessing import load_data, clean_data, scale_data
from src.clustering import run_clustering
from src.evaluation import evaluate_clustering
from src.utils import plot_correlation_heatmap, get_correlation_matrix

def main():
    # Step 1: Load the data
    data = load_data('data/raw/dataset.xlsx')

    # Step 2: Clean and preprocess the data
    cleaned_data = clean_data(data)
    scaled_data = scale_data(cleaned_data)

    # Step 3: Generate and plot the correlation matrix
    corr_matrix = get_correlation_matrix(scaled_data)
    plot_correlation_heatmap(corr_matrix)

    # Step 4: Run the clustering algorithm
    clusters = run_clustering(scaled_data)

    # Step 5: Evaluate the clustering performance
    evaluate_clustering(clusters, scaled_data)

if __name__ == '__main__':
    main()
