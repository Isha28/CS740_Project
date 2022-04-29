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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt  
import flow_stats
import time
import bagwords


bag_of_words_df = bagwords.generate_bag_of_words("traces/output_sample.csv")

labels = pd.DataFrame(bag_of_words_df.index)
data = bag_of_words_df

x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=1 )

multinomial_nb =  MultinomialNB()
multinomial_nb.fit(x_train, y_train)

y_preds = multinomial_nb.predict(x_test)

print(y_preds[:15])
print(y_test[:15])

print('Test Accuracy : %.3f'%multinomial_nb.score(x_test, y_test)) ## Score method also evaluates accuracy for classification models.
print('Training Accuracy : %.3f'%multinomial_nb.score(x_train, y_train))

