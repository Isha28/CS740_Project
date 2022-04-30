import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
import six
import joblib
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
import os


def train_bag_of_words(df,model_path, flow_id_device_map):
    df["Class"] = df.index.map(flow_id_device_map)
    df.dropna(subset =["Class"],inplace=True)
    labels = df.loc[:,"Class"]
    data = df.loc[:,df.columns != "Class"]
    x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=1 )
    multinomial_nb =  MultinomialNB()
    multinomial_nb.fit(x_train, y_train)
    joblib.dump(multinomial_nb, model_path + ".joblib")
    #Evaluation
    y_preds = multinomial_nb.predict(x_test)
    class_probabilities = multinomial_nb.predict_proba(x_test)
    # print(y_preds)
    # print([max(probabilities) for probabilities in class_probabilities])

    print('Test Accuracy : %.3f'%multinomial_nb.score(x_test, y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%multinomial_nb.score(x_train, y_train))


def predict_bag_of_words(model_path, filename, attribute=None):
    bow_dfs = bagwords.generate_bag_of_words(filename,attribute)
    data = bow_dfs[1]
    print("Bagwords returns " + str(len(data)) + " rows")
    # print(data)
    model = joblib.load(model_path)
    preds = model.predict(data)
    # print(preds)
    prob_values = [max(probabilities) for probabilities in model.predict_proba(data)]
    class_column = attribute + "_class"
    confidence_column = attribute + "_confidence"
    output_file = filename.split('.')[0] + "_stage0.csv"
    if os.path.exists(output_file):
        new_data = pd.read_csv(output_file)
    else:
        new_data = pd.read_csv(filename)
    print("New data has " + str(len(new_data)) + " rows")
    new_data[class_column] = preds
    new_data[confidence_column] = prob_values
    new_data.to_csv(output_file)

def predict_all(filename):
    bow_dfs = bagwords.generate_bag_of_words(filename)

    flow_id_device_map = bow_dfs[0]
    remote_ports_df = bow_dfs[1]
    domains_df = bow_dfs[2]
    train_bag_of_words(remote_ports_df, "stage0_ports",flow_id_device_map)
    train_bag_of_words(domains_df, "stage0_domains",flow_id_device_map)
    predict_bag_of_words("stage0_ports.joblib",filename,attribute="ports")
    predict_bag_of_words("stage0_domains.joblib",filename,attribute="domains")

if __name__ == '__main__':
    predict_all("traces/output_sample.csv")




