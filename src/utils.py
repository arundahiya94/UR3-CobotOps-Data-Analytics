import seaborn as sns
import matplotlib.pyplot as plt

def get_correlation_matrix(data):
    """Calculate the correlation matrix."""
    return data.corr()

def plot_correlation_heatmap(corr_matrix, figsize=(20, 10)):
    """Plot a heatmap of the correlation matrix."""
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
