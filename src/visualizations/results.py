import sys
sys.path.append("../src")   # add src/ to Python path

import pandas as pd
import matplotlib.pyplot as plt

from visualizations.exploratory import save_figure


def plot_scores(scores, filename='models_scores.png'):
  compare = pd.DataFrame(scores, index=["accuracy"])
  ax = compare.T.plot.bar(legend=False)
  plt.xticks(rotation=0)
  plt.ylabel("Score")

  for container in ax.containers:
    ax.bar_label(container, label_type="edge")

  save_figure(filename)
  plt.show()
  plt.close()
