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
import joblib

from sklearn.metrics import ConfusionMatrixDisplay

def knn_classifier(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=28)
    knn = Pipeline([('norm',StandardScaler()),('knn', knn)])
    knn.fit(x_train,y_train)
    pred_values = knn.predict(x_test)
    print(pred_values)
    joblib.dump(knn, "knn_stage_1" + ".joblib")
    #Evaluation
    class_probabilities = knn.predict_proba(x_test)
    
    print('Test Accuracy : %.3f'%knn.score(x_test, y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%knn.score(x_train, y_train))
    
    print (classification_report(y_test, pred_values))

    class_names = knn.classes_ 
    plt.rcParams.update({'font.size': 4})
    cm = confusion_matrix(y_test,pred_values,labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.show()
   
    return pred_values

def random_forest_classifier(x_train, y_train, x_test, y_test):
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(x_train,y_train)
    pred_values=clf.predict(x_test)
    print(pred_values)
    
    joblib.dump(clf, "random_forest_stage_1" + ".joblib")
    #Evaluation
    class_probabilities = clf.predict_proba(x_test)
    
    print('Test Accuracy : %.3f'%clf.score(x_test, y_test))
    print('Training Accuracy : %.3f'%clf.score(x_train, y_train))
    
    print (classification_report(y_test, pred_values))

    class_names = clf.classes_ 
    plt.rcParams.update({'font.size': 4})
    cm = confusion_matrix(y_test,pred_values,labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.show()
    
    return pred_values

def svm_classifier(x_train, y_train, x_test, y_test):
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    print(pred_values)
    
    joblib.dump(clf, "svm_stage_1" + ".joblib")
    #Evaluation
    class_probabilities = clf.predict_proba(x_test)
    
    print('Test Accuracy : %.3f'%clf.score(x_test, y_test))
    print('Training Accuracy : %.3f'%clf.score(x_train, y_train))
    
    print (classification_report(y_test, pred_values))

    class_names = clf.classes_ 
    plt.rcParams.update({'font.size': 4})
    cm = confusion_matrix(y_test,pred_values,labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.show()
    
    return pred_values


def main():
    filename = "merged_traces_unsw_2_updated_stage0.csv"
    df = pd.read_csv(filename)
    print(df)

    data = df.loc[:,["flow_duration","flow_rate", "flow_volume", "sleep_time","dest_port",\
    "domains_class","domains_confidence","cipher_suites_class","cipher_suites_confidence"]]
    print(data)
    data.fillna(0,inplace=True)
    labels = df.loc[:,["Class"]]
    print("Labels ", labels)

    t3 = time.time()
    x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=1 )

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

    pred_values = knn_classifier(x_train, y_train, x_test, y_test)

    print(pred_values)
    t4 = time.time()
    print("Prediction took " + str(t4-t3) + " seconds")

if __name__=="__main__":
    main()



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




