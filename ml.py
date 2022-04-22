import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt  
import flow_stats
import time

filename = "sample2.csv"
df = pd.read_csv(filename)
print(df)
data = df.iloc[:,[3,4,5]]
print(data)
labels = df.iloc[:,[1]]
print(labels)
t3 = time.time()
x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.4, random_state=1 )
knn = KNeighborsClassifier(n_neighbors=30)
knn = Pipeline([('norm',StandardScaler()),('knn', knn)])
knn.fit(x_train,y_train)
pred_values = knn.predict(x_test)
t4 = time.time()
print("Prediction took " + str(t4-t3) + " seconds")
print(pred_values)