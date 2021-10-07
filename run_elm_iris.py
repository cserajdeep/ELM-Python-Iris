import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

np.random.seed(42)

class ELM(object):  
    
  def __init__(self, inputSize, outputSize, hiddenSize):
    """
    Initialize weight and bias between input layer and hidden layer
    Parameters:
    inputSize: int
        The number of input layer dimensions or features in the training data
    outputSize: int
        The number of output layer dimensions
    hiddenSize: int
        The number of hidden layer dimensions        
    """    

    self.inputSize = inputSize
    self.outputSize = outputSize
    self.hiddenSize = hiddenSize       
    
    # Initialize random weight with range [-0.5, 0.5]
    self.weight = np.matrix(np.random.uniform(-0.5, 0.5, (self.hiddenSize, self.inputSize)))

    # Initialize random bias with range [0, 1]
    self.bias = np.matrix(np.random.uniform(0, 1, (1, self.hiddenSize)))
    
    self.H = 0
    self.beta = 0

  def sigmoid(self, x):
    """
    Sigmoid activation function
    
    Parameters:
    x: array-like or matrix
        The value that the activation output will look for
    Returns:      
        The results of activation using sigmoid function
    """
    return 1 / (1 + np.exp(-1 * x))

  def predict(self, X):
    """
    Predict the results of the training process using test data
    Parameters:
    X: array-like or matrix
        Test data that will be used to determine output using ELM
    Returns:
        Predicted results or outputs from test data
    """
    X = np.matrix(X)
    y = self.sigmoid((X * self.weight.T) + self.bias) * self.beta

    return y

  def train(self, X, y):
    """
    Extreme Learning Machine training process
    Parameters:
    X: array-like or matrix
        Training data that contains the value of each feature
    y: array-like or matrix
        Training data that contains the value of the target (class)
    Returns:
        The results of the training process   
    """

    X = np.matrix(X)
    y = np.matrix(y)        
    
    # Calculate hidden layer output matrix (Hinit)
    self.H = (X * self.weight.T) + self.bias

    # Sigmoid activation function
    self.H = self.sigmoid(self.H)

    # Calculate the Moore-Penrose pseudoinverse matriks        
    H_moore_penrose = np.linalg.inv(self.H.T * self.H) * self.H.T

    # Calculate the output weight matrix beta
    self.beta = H_moore_penrose * y

    return self.H * self.beta

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
  """
  Generate matrix plot of confusion matrix with pretty annotations.
  The plot image is saved to disk.
  args: 
    y_true:    true label of the data, with shape (nsamples,)
    y_pred:    prediction of the data, with shape (nsamples,)
    filename:  filename of figure file to save
    labels:    string array, name the order of class labels in the confusion matrix.
                use `clf.classes_` if using scikit-learn models.
                with shape (nclass,).
    ymap:      dict: any -> string, length == nclass.
                if not None, map the labels & ys to more understandable strings.
                Caution: original y_true, y_pred and labels must align.
    figsize:   the size of the figure plotted.
  """
  if ymap is not None:
      y_pred = [ymap[yi] for yi in y_pred]
      y_true = [ymap[yi] for yi in y_true]
      labels = [ymap[yi] for yi in labels]
  cm = confusion_matrix(y_true, y_pred)
  cm_sum = np.sum(cm, axis=1, keepdims=True)
  cm_perc = cm / cm_sum.astype(float) * 100
  annot = np.empty_like(cm).astype(str)
  nrows, ncols = cm.shape
  for i in range(nrows):
      for j in range(ncols):
          c = cm[i, j]
          p = cm_perc[i, j]
          if i == j:
              s = cm_sum[i]
              annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
          elif c == 0:
              annot[i, j] = ''
          else:
              annot[i, j] = '%.1f%%\n%d' % (p, c)
  
  cm = pd.DataFrame(cm, index=labels, columns=labels)
  cm.index.name = 'Actual'
  cm.columns.name = 'Predicted'
  fig, ax = plt.subplots(figsize=figsize)
  sns.heatmap(cm, annot=annot, cmap="YlGnBu", fmt='', ax=ax, linewidths=.5)
  #plt.savefig(filename)
  plt.show()

# Create random classification datasets with 1000 samples
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create instance of ELM object with 10 hidden neuron
elm = ELM(X.shape[1], 1, 10)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=True)
print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

# Train data
elm.train(X_train,y_train.reshape(-1,1))

# Make prediction from training process
y_pred = elm.predict(X_test)
y_pred = np.round(np.abs(y_pred))
print(y_test, "\n", y_pred.T)

print("\nAccuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
print("Recall Score: ", recall_score(y_test, y_pred, average="macro")) 
print("\n", classification_report(y_test, y_pred, target_names=names))

cm_analysis(y_test, y_pred, names, ymap=None, figsize=(6,6))
cm = confusion_matrix(y_test, y_pred, normalize='all')
cmd = ConfusionMatrixDisplay(cm, display_labels=names)
cmd.plot()
cmd.ax_.set(xlabel='Predicted', ylabel='True')
#plt.savefig("Confusion_Matrix.png")