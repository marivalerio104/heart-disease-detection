import pandas as pd

def split_features_target(df: pd.DataFrame):
  X = df.drop(columns=['target'])
  y = df.target
  return X, y
