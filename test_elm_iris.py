from elm import ELM
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

np.random.seed(42)

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
print('Accuracy: ', accuracy_score(y_test, y_pred))

print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
print("Recall Score: ", recall_score(y_test, y_pred, average="macro")) 
print(classification_report(y_test, y_pred, target_names=names))