import joblib


def load_trained_models():
  models = {"Logistic_Regression": joblib.load("../models/Logistic_Regression.pkl"),
            "KNN": joblib.load("../models/KNN.pkl"),
            "Random_Forest": joblib.load("../models/Random_Forest.pkl")}

  return models


def score_models(models, X_test, y_test):
  '''
  Scores the given models
  
  Parameters
  ----------
  models : dictionary
    Models to score.
  X_test : pd.DataFrame
    Testing data with no labels.
  y_test : pd.Series
    Testing labels.

  Returns
  -------
  scores : dictionary
  '''

  # Disctionary to keep model scores
  scores = {}

  # Loop through the models
  for name, model in models.items():
    scores[name] = model.score(X_test, y_test)
  
  return scores
  