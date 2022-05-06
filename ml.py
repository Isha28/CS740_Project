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

def random_forest_validation(X,Y,cvol=True,k=5):
    models = []
    acc_score = []
    print("Types:")
    print(type(X))
    print(type(Y))
    model = RandomForestClassifier(n_estimators=100)
    if cvol: 
        kf = KFold(n_splits=k, random_state=None)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = Y.iloc[train_index] , Y.iloc[test_index]
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)
            # pred_values = model.predict_proba(X_test)
            # predicted = (pred_values [:,1] >= threshold).astype('int')
            acc = accuracy_score(y_test , pred_values)
            print("Training: %d", acc)
            acc_score.append(acc)
            models.append(model)
    else:
        model = RandomForestClassifier(n_estimators=10,max_depth=3)
        model.fit(X,Y)
        models.append(model)
    
    return models, acc_score


def random_forest_classifier(x_train, y_train, x_test, y_test, cvol=True,k=5):
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

def predict_random_forest_classifiers(X_test,train_columns,test_columns,Y_test):
    train_feature = train_columns.intersection(test_columns)
    dummy_columns = train_columns.difference(test_columns)
    X_test = X_test[train_feature]
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    print("Test feature vector length: ", len(test_columns))
    print("Intersection: ", len(train_feature))
    print("Dummy: ", len(dummy_columns))
    # r = X_test.index[np.isnan(X_test).any(1)]
    # print(r)
    # print(len(X_test))
    # # print(X_test.iloc[380])
    model = joblib.load('random_forest_stage_1.joblib')
    pred_values = model.predict(X_test)
    acc = accuracy_score(Y_test , pred_values)
    # threshold = 0.5
    # pred_values = model.predict_proba(X_test)
    # predicted = [model.classes_[np.where(p==max(p))][0] for p in pred_values]
    # acc = accuracy_score(Y_test["Class"].tolist() , predicted)
    print("Training: ", acc)
    
    # return predicted

def svm_classifier(x_train, y_train, x_test, y_test):
    clf = svm.SVC() # Linear Kernel
    clf.fit(x_train, y_train)
    pred_values = clf.predict(x_test)
    print(pred_values)
    
    joblib.dump(clf, "svm_stage_1" + ".joblib")
    #Evaluation
    
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
    filename = "merged_traces_lab2_updated.csv"
    df = pd.read_csv(filename)
    print(df)
    test_columns = {"flow_duration","flow_rate", "flow_volume", "sleep_time","dest_port"}
    data = df.loc[:,list(test_columns)]
    # print(data)
    data.fillna(0,inplace=True)
    labels = df.loc[:,["Class"]]
    print("Labels ", labels)
    # pred_values = predict_random_forest_classifiers(data, test_columns, test_columns, labels)
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

    pred_values = predict_random_forest_classifiers(data, test_columns, test_columns, labels)
    # models, _ = random_forest_validation(x_train, y_train)
    # pred_values = knn_classifier(x_train, y_train, x_test, y_test)
    # pred_values = random_forest_classifier(x_train, y_train, x_test, y_test)
    # pred_values = svm_classifier(x_train, y_train, x_test, y_test)
    
    # print(pred_values)
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




