import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
  return pd.read_csv(filepath)


def clean_data(df) -> pd.DataFrame:
  # Drop non important column
  df = df.drop(columns=['patientid'])

  folder_path = '../data/processed'
  # Create the folder if it does not exist
  os.makedirs(folder_path, exist_ok=True)
  file_path = os.path.join(folder_path, 'heart-disease.csv')
  df.to_csv(file_path, index=False)

  return df
