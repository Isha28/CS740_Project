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
import pickle

def read_data(inputcsv):
    data = pd.read_csv(inputcsv,index_col=False)
    # del train_data["app_id"]
    label = data["Class"] # Target variable
    del data["Class"]
    # del data["mapping"]
    #app_id = data.appID
    #del data["appID"]
    columns = data.columns
    return data, label, columns

def train_bag_of_words(df,model_path, flow_id_device_map):
    print("Beginning training")
    df["Class"] = df.index.map(flow_id_device_map)
    df.dropna(subset =["Class"],inplace=True)
    labels = df.loc[:,"Class"]
    print("Extracted Labels")
    data = df.loc[:,df.columns != "Class"]
    print("Extracted Data")
    print(data)
    print("Number of columns:" + str(len(data.columns)))
    with open(model_path+"train_columns.pkl","wb") as f:
        pickle.dump(data.columns, f)
   
    x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=1)
    multinomial_nb =  MultinomialNB()
    multinomial_nb.fit(x_train, y_train)
    # print("Split the data into chunks")
    # split_data = np.array_split(data,3)
    # for single_data in split_data:
    #     print("training on new chunk")
    #     x_train, x_test, y_train, y_test = train_test_split(single_data,labels, test_size=0.25, random_state=1 )
    #     multinomial_nb =  MultinomialNB()
    #     multinomial_nb.partial_fit(x_train, y_train)
    joblib.dump(multinomial_nb, model_path + ".joblib")
    #Evaluation
    y_preds = multinomial_nb.predict(x_test)
    print(y_preds)
    class_probabilities = multinomial_nb.predict_proba(x_test)
    # print(y_preds)
    # print([max(probabilities) for probabilities in class_probabilities])

    print('Test Accuracy : %.3f'%multinomial_nb.score(x_test, y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%multinomial_nb.score(x_train, y_train))


def train_hourly_bag_of_words(df,model_path):
    print("Beginning training")
    df.dropna(subset =["Class"],inplace=True)
    df.fillna(0,inplace=True)
    labels = df.loc[:,"Class"]
    print("Extracted Labels")
    print(labels)
    data = df.loc[:,(df.columns != "Class") & (df.columns != "mapping")]
    data = data.astype(int)
    print("Extracted Data")
    print(data)
    with open(model_path+"train_columns.pkl","wb") as f:
        pickle.dump(df.columns, f)
    x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=1)
    multinomial_nb =  MultinomialNB()
    multinomial_nb.fit(x_train, y_train)
    # print("Split the data into chunks")
    # split_data = np.array_split(data,3)
    # for single_data in split_data:
    #     print("training on new chunk")
    #     x_train, x_test, y_train, y_test = train_test_split(single_data,labels, test_size=0.25, random_state=1 )
    #     multinomial_nb =  MultinomialNB()
    #     multinomial_nb.partial_fit(x_train, y_train)
    joblib.dump(multinomial_nb, model_path + ".joblib")
    #Evaluation
    y_preds = multinomial_nb.predict(x_test)
    print(y_preds)
    class_probabilities = multinomial_nb.predict_proba(x_test)
    # print(y_preds)
    # print([max(probabilities) for probabilities in class_probabilities])

    print('Test Accuracy : %.3f'%multinomial_nb.score(x_test, y_test)) ## Score method also evaluates accuracy for classification models.
    print('Training Accuracy : %.3f'%multinomial_nb.score(x_train, y_train))

def bucketize_flow_id_to_hourly_instance(mapping_dict):
    bucket_dict = {}
    for instance_id in mapping_dict:
        for flow_id in mapping_dict[instance_id]:
            bucket_dict[flow_id] = instance_id
    return bucket_dict

def predict_hourly_bag_of_words(model_path, test_columns, bag_of_words_file, filename, attribute=None):
    df = pd.read_csv(bag_of_words_file)
    print("bag of words file read complete")
    # df.dropna(subset =["class"],inplace=True)
    df.fillna('0',inplace=True)
    mapping_dict = {df.index[idx]:df.loc[idx,"mapping"].split('|') for idx in range(len(df))}
    print("mapping dict created")
    flow_to_instance_dict = bucketize_flow_id_to_hourly_instance(mapping_dict)
    print(mapping_dict)
    data = df.loc[:,df.columns != "class"]
    X_test = data.drop("mapping",1)
    print("X_test extracted")
    with open(model_path + "train_columns.pkl", 'rb') as f:
        train_columns = pickle.load(f)
    train_feature = train_columns.intersection(data)
    dummy_columns = train_columns.difference(data)
    X_test = X_test[train_feature]
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    print("Extracted Data")
    print(data)
    print("Running classifier on " + str(len(data)) + " rows")
    # print(data)
    model = joblib.load(model_path)
    preds = model.predict(data)
    # print(preds)
    prob_values = [max(probabilities) for probabilities in model.predict_proba(data)]
    print("Prediction successful")
    class_column = attribute + "_class"
    confidence_column = attribute + "_confidence"
    output_file = filename.split('.')[0] + "_stage0.csv"
    if os.path.exists(output_file):
        new_data = pd.read_csv(output_file)
    else:
        new_data = pd.read_csv(filename)
    print(new_data)
    # for idx, row in new_data.iterrows():
    #     new_data.loc[idx][class_column] = 
    print("New data has " + str(len(new_data)) + " rows")
    new_data[class_column] = preds
    new_data[confidence_column] = prob_values
    new_data.to_csv(output_file)

def predict_bag_of_words(model_path, X_test,test_columns, filename, attribute=None):
    # df = pd.read_csv(bag_of_words_file)
    # print("bag of words file read complete")
    # print(data)
    X_test.fillna('0',inplace=True)
    X_test= X_test.loc[:,X_test.columns != "Class"]
    print(X_test)
    print(X_test.columns)
    print("X_test extracted")
    with open(model_path.split('.')[0] + "train_columns.pkl", 'rb') as f:
        train_columns = pickle.load(f)
    print(len(train_columns))
    train_feature = train_columns.intersection(X_test)
    dummy_columns = train_columns.difference(X_test)
    X_test = X_test[train_feature]
    X_test = X_test.reindex(X_test.columns.union(dummy_columns, sort=False), axis=1, fill_value=0)
    X_test = X_test[train_columns]
    print(X_test)
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    # print(preds)
    prob_values = [max(probabilities) for probabilities in model.predict_proba(X_test)]
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
    # bow_dfs = bagwords.generate_bag_of_words(filename, attribute="cipher_suites")
    X_test, Y_test, test_columns = read_data(filename)
    print("data read complete")
    predict_bag_of_words("stage0_cipher_suites_flows.joblib",X_test,test_columns,filename,attribute="cipher_suites")

    # flow_id_device_map = bow_dfs[0]
    # cipher_suites_df = bow_dfs[1]
    # # print(cipher_suites_df)
    # train_bag_of_words(cipher_suites_df, "stage0_cipher_suites_flows",flow_id_device_map)
    # remote_ports_df = bow_dfs[1]
    # domains_df = bow_dfs[1]
    # train_hourly_bag_of_words(remote_ports_df)
    # train_bag_of_words(remote_ports_df, "stage0_ports_flows",flow_id_device_map)
    # train_bag_of_words(domains_df, "stage0_domains_flows",flow_id_device_map)
    # predict_hourly_bag_of_words("stage_0_ports_hourly.joblib",test_columns, \
        # "merged_bag_of_words_ports2.csv",filename,attribute="ports")
    # predict_hourly_bag_of_words("stage_0_domains_hourly.joblib","merged_bag_of_words_domains_test.csv",filename,attribute="domains")
    # predict_bag_of_words("stage0_domains_flows.joblib",test_columns, "merged_bag_of_words_domains.csv",filename,attribute="domains")
    # predict_bag_of_words("stage0_cipher_suites_flows.joblib",filename,attribute="cipher_suites")

def predict_hourly_instances(filename):
    df = pd.read_csv(filename,index_col=False)
    train_hourly_bag_of_words(df, "stage_0_ports_hourly")

if __name__ == '__main__':
    predict_all("merged_traces_unsw_2_updated.csv")
    # predict_hourly_instances("merged_bag_of_words_ports2.csv")



