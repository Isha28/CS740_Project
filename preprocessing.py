import os
import pandas as pd
import numpy as np
import flow_stats
import time
import argparse
import os
import csv

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

def filter_output(filename):
    with open("maclist.txt") as f:
        target_macs = [line.rstrip().lower() for line in f]
    df = pd.read_csv(filename,index_col=0)
    df = df[df['source_mac'].isin(target_macs) | df["dest_mac"].isin(target_macs)]
    df.to_csv(filename.split('.')[0]+"_filtered.csv")


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
        merge_files(dirname, "merged_traces_unsw.csv")
    elif (args.filename):
        filename = args.filename
        filter_output(filename)

if __name__ == "__main__":
    main()
