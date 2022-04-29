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
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_roc_curve
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt  
import flow_stats
import time

filename = "traces/output_sample.csv"
df = pd.read_csv(filename)
print(df)

data = df.iloc[:,[1,2,3,4,7,8,9]]
print(data)

labels = df.iloc[:,[6]]
print(labels)

t3 = time.time()
x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.4, random_state=1 )

le = preprocessing.LabelEncoder()
for column_name in x_train.columns:
    if x_train[column_name].dtype == object:
        x_train[column_name] = le.fit_transform(x_train[column_name])
    else:
        pass
for column_name in x_test.columns:
    if x_test[column_name].dtype == object:
        x_test[column_name] = le.fit_transform(x_test[column_name])
    else:
        pass

#knn
knn = KNeighborsClassifier(n_neighbors=30)
knn = Pipeline([('norm',StandardScaler()),('knn', knn)])
knn.fit(x_train,y_train)
pred_values = knn.predict(x_test)

#svm
#clf = svm.SVC(kernel='linear') # Linear Kernel
#clf.fit(x_train, y_train)
#pred_values = clf.predict(x_test)

#random forest
#clf=RandomForestClassifier(n_estimators=100)
#clf.fit(x_train,y_train)
#pred_values=clf.predict(x_test)

#confusion_matrix(y_test,pred_values)
#pd.crosstab(y_test, pred_values, rownames = ['Actual'], colnames =['Predicted'], margins = True)
print(classification_report(y_test, pred_values))

t4 = time.time()
print("Prediction took " + str(t4-t3) + " seconds")
print(pred_values)
pred_dict = {}
for a in pred_values:
    if a not in pred_dict:
        pred_dict[a] = 1
    else:
        pred_dict[a] += 1
print ("Occurence of each prediction", pred_dict)
