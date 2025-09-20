import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def save_figure(filename):
  folder_path = '../reports/figures'
  # Create the folder if it does not exist
  os.makedirs(folder_path, exist_ok=True)
  save_path = os.path.join(folder_path, filename)
  plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_target_vs_sex(df, filename='target_vs_sex.png'):
  ct = pd.crosstab(df.target, df.sex)
  ax = ct.plot(kind='bar', rot=0)
  
  # Customize
  ax.set_title('Target vs Sex')
  ax.legend(['Female', 'Male'], title='Sex')
  ax.set_xlabel('Target')
  ax.set_ylabel('Count')
  ax.set_xticklabels(['No disease', 'Disease'])
  
  # Add counts on top of each bar
  for container in ax.containers:
    ax.bar_label(container, label_type='edge')

  save_figure(filename)
  
  plt.show()
  plt.close()


def plot_age_vs_thalach(df, filename='age_vs_thalach.png'):
  plt.figure()
  # Plot for no disease
  plt.scatter(df.age[df.target==0], df.thalach[df.target==0])
  # Plot for disease
  plt.scatter(df.age[df.target==1], df.thalach[df.target==1])

  plt.title('Age vs Max Heart Rate by Disease')
  plt.legend(['No disease', 'Disease'])
  plt.xlabel('Age')
  plt.ylabel('Max Heart Rate')

  save_figure(filename)
  
  plt.show()
  plt.close()


def plot_age_distribution(df, filename='age_distribution.png'):
  plt.figure()
  plt.hist([df.age[df.target==0], df.age[df.target==1]])

  plt.title('Distribution of Age by Disease')
  plt.xlabel('Age')
  plt.legend(['No disease', 'Disease'])

  save_figure(filename)

  plt.show()
  plt.close()


def plot_correlation_matrix(df, filename='correlation_matrix.png'):
  corr_matrix = df.corr()

  plt.figure(figsize=(10, 7))
  sns.heatmap(corr_matrix, linewidths=0.5, annot=True, fmt='.2f', cmap='coolwarm')
  plt.title('Correlation Matrix', fontsize=14)

  save_figure(filename)

  plt.show()
  plt.close()
