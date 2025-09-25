import joblib
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(X: pd.DataFrame, y: pd.Series,
  test_size: float = 0.2, random_state: int = 42, 
  stratify: bool = True
):
  '''
  Split features and target into train and test sets, and save them as CSVs. 
  
  Parameters
  ----------
  X : pd.DataFrame
    Feature matrix.
  y : pd.Series
    Target vector.
  test_size : float, default=0.2
    Proportion of dataset to include in test split.
  random_state : int, default=42
    Random seed for reproducibility.
  stratify : bool, default=True
    Whether to stratify the split by y .
  
  Returns
  -------
  X_train, X_test, y_train, y_test
    Train/test splits.
  '''

  stratify_param = y if stratify else None

  # Split the data
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=test_size, random_state=random_state, stratify=stratify_param
  )

  # Concat training sets and testing sets
  train_df = pd.concat([X_train, y_train], axis=1)
  test_df = pd.concat([X_test, y_test], axis=1)

  # Save the sets
  os.makedirs('../data/processed', exist_ok=True)
  train_df.to_csv('../data/processed/train.csv', index=False)
  test_df.to_csv('../data/processed/test.csv', index=False)
  
  return X_train, X_test, y_train, y_test


def train_models(models, X_train, y_train):
  '''
  Trains the given models and saves the trained models
  
  Parameters
  ----------
  models : dictionary
    Models to train.
  X_train : pd.DataFrame
    Training data with no labels.
  y_train : pd.Series
    Training labels.
  '''

  np.random.seed(42)
  os.makedirs('../models', exist_ok=True)

  # Loop through the models
  for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    # Save the model
    model_path = "../models/" + name + ".pkl"
    joblib.dump(model, model_path)
