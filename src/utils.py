from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def perform_pca(df, n_components=2, plot_pca=True):
    # Ensure only numeric columns are used
    df_numeric = df.select_dtypes(include=[float, int])
    
    # Check if DataFrame is empty after selecting numeric columns
    if df_numeric.empty:
        raise ValueError("The DataFrame doesn't contain any numeric data to perform PCA.")
    
    # Handle missing values (optional, can be customized)
    if df_numeric.isnull().any().any():
        df_numeric = df_numeric.fillna(df_numeric.mean())  # You can also use df_numeric.dropna()
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df_numeric)
    
    if plot_pca:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7, cmap='viridis')
        plt.title('PCA - 2 Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    return pca_data

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
