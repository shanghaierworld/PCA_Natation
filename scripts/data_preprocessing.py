# Script for data preprocessing
import pandas as pd

def preprocess_data(file_path):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    # Add preprocessing steps here
    return data