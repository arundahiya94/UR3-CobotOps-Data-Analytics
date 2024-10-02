import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from file."""
    return pd.read_excel(file_path)

def clean_data(data):
    """Clean the dataset: handle missing values and remove unwanted columns."""
    data.dropna(inplace=True)
    data.drop('Timestamp', axis=1, inplace=True)
    return data

def scale_data(data):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return scaled_data
