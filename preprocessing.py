import os
import pandas as pd
import numpy as np
import flow_stats
import time
import argparse
import os
import csv
import math
from subprocess import Popen


def process_csv(filename):
    example_file = open(filename,encoding='utf-8')
    example_reader = csv.reader(example_file)
    example_data = list(example_reader)
    example_file.close()
    return example_data

def merge_files(dirname, output_path):
    files = []
    for path in os.listdir(dirname):
        if (path.split('.')[-1] == "csv"):
            files.append(os.path.abspath(dirname + path))
    merged_rows = []
    t_count = 1
    u_count = 1
    print(files)
    for idx, single_file in enumerate(files):
        print(single_file)
        rows = process_csv(single_file)
        for row in rows[1:]:
            if row[0][0] == 't':
                row[0] = 't' + str(t_count)
                t_count +=1 
            elif row[0][0] == 'u':
                row[0] = 'u' + str(u_count)
                u_count +=1 
        if idx == 0:
            merged_rows.extend(rows)
        else:
            merged_rows.extend(rows[1:])
    df = pd.DataFrame(merged_rows)
    print(df)
    df.columns = df.iloc[0]
    df.drop(df.index[0],inplace=True)
    df.set_index(df.iloc[:, 0], inplace=True)
    df.to_csv(output_path,index=False)

def split_data_by_hour(filename):
    df = pd.read_csv(filename)
    df = df[(df["start_time"] >= 1474552802) & (df["end_time"] >= 1474552802)]
    df = df.sort_values(by="start_time")
    df.to_csv("sorted_merged_values.csv",index=False)
    # print(df)
    begin_time = df.iloc[0]["start_time"]
    print(begin_time)
    end_time = df.iloc[-1]["end_time"]
    print(end_time)
    num_hours = math.ceil((end_time - begin_time)/3600)
    # print(num_hours)
    all_dfs = []
    for idx in range(num_hours):
        print(idx)
        new_df = df[(df["start_time"] >= begin_time + 3600*(idx)) & (df["start_time"] < begin_time + 3600*(idx+1))]
        # print(new_df)
        all_dfs.append(new_df)
    print("Computed list of dataframes")
    for idx, single_df in enumerate(all_dfs):
        if single_df.empty:
            continue
        single_df.to_csv("hourly_output/hour_" + str(idx) + ".csv")

def merge_pcaps_by_hour(dirname,output_dir):
    file_list = sorted(os.listdir(dirname))
    file_list = [str(dirname + path) for path in file_list if path[-4:] == 'pcap']
    for idx, path in enumerate(file_list):
        if (idx % 4 == 0):
            process = Popen(['mergecap', '-w', output_dir+ "/merged_4_" + str(idx), file_list[idx], file_list[idx+1], file_list[idx+2] \
                ,file_list[idx+3]], shell=False, close_fds=True)
            stdout, stderr = process.communicate()
        
        

def filter_output(filename):
    with open("maclist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    df = pd.read_csv(filename,index_col=0)
    df = df[df['source_mac'].isin(target_macs) | df["dest_mac"].isin(target_macs)]
    df.to_csv(filename.split('.')[0]+"_filtered.csv")

def replace_mac_with_device_name(filename, column_name):
    df = pd.read_csv(filename)
    print(df)
    with open("devicelist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    print(target_macs)
    device_map = {}
    for mac in target_macs:
        device_mac = mac.split("-")[0]
        device_name = mac.split("-")[1]
        device_map[device_mac] = device_name
    
    print ("Device mappings", device_map)
    df = df[df[column_name].isin(device_map)]
    df.rename(columns={column_name:"Class"},inplace=True)
    df["Class"] = df["Class"].apply(lambda x: device_map[x])
    df.to_csv(filename.split('.')[0] + "_updated.csv")



def generate_mapping_dict(bag_of_words_file):
    for idx in range(25):
        df = pd.read_csv(bag_of_words_file, skiprows = idx*10000, nrows=(idx+1)*10000)
        print("bag of words file read for idx " + str(idx) + " complete")
        # df.dropna(subset =["class"],inplace=True)
        df.fillna('0',inplace=True)
        mapping_dict = {df.index[index]:df.loc[index,"mapping"].split('|') for index in range(len(df))}
        print("mapping dict created")
        with open("mapping_dict_"+str(idx),"wb") as f:
            pickle.dump(mapping_dict,f)

def args_parser():
    parser = argparse.ArgumentParser(description="Specify CSV files to merge")
    parser.add_argument("-d","--directory",dest="dirname",help="directory of csv files")
    parser.add_argument("-f","--filename",dest="filename",help="file to filter")
    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    if(args.dirname):
        dirname = args.dirname
        merge_files(dirname, "merged_traces_lab_unsw.csv")
    elif (args.filename):
        filename = args.filename
        filter_output(filename)

if __name__ == "__main__":
    # split_data_by_hour("merged_traces_unsw_2.csv")
    # merge_pcaps_by_hour("lab/lab4/","lab/merged/")
    # main()
    replace_mac_with_device_name("merged_traces_lab_unsw.csv","dest_mac")
    # generate_mapping_dict("merged_bag_of_words_ports2.csv")
